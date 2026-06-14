"""PUCT Monte Carlo Tree Search -- the heart of AlphaZero.

A faithful-but-tiny version of raccoon's ``raccoon/search/mcts.py``. Each
simulation:

1. **Select** -- walk down the tree, at each *decision* node picking the child
   that maximises the PUCT score

       score(a) = Q_parent(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))

   where ``Q_parent`` is the child's value *from the parent's point of view*
   (negated when the child belongs to the opponent). At a *chance* node we
   instead **sample** an outcome from its probabilities -- the dice are sampled
   and the node skipped, never handed to the network (raccoon's design).
2. **Expand** -- evaluate a freshly reached decision leaf with the network to get
   priors P(a) and a value v.
3. **Back up** -- add v along the path, flipping its sign whenever the node's
   player differs from the leaf's player (zero-sum negamax backup; this single
   rule handles chance nodes correctly because they don't change the player).

The visit counts at the root become the policy target -- a "search-refined"
distribution that is a little smarter than the raw network prior, which is what
makes the whole self-play bootstrap work.
"""

from __future__ import annotations

import math

import numpy as np

from azl.games.base import GameState
from azl.network import AZNet


class Node:
    __slots__ = ("state", "parent", "prior", "children", "child_priors",
                 "visit_count", "value_sum", "is_expanded")

    def __init__(self, state: GameState, parent: "Node | None" = None, prior: float = 0.0):
        self.state = state
        self.parent = parent
        self.prior = prior
        self.children: dict[int, Node] = {}
        self.child_priors: dict[int, float] = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Mean value from *this node's* player's perspective."""
        return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count


class MCTS:
    def __init__(
        self,
        net: AZNet,
        num_simulations: int = 50,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.0,
        noise_eps: float = 0.25,
        rng: np.random.Generator | None = None,
    ):
        self.net = net
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.noise_eps = noise_eps
        self.rng = rng or np.random.default_rng()

    # --- public API -------------------------------------------------------
    def search(self, state: GameState, add_noise: bool = True):
        """Run simulations from ``state`` (must be a decision node).

        Returns ``(visit_probs, root_value, visit_counts)``:
            visit_probs  -- dict action -> normalised visit count (the policy target)
            root_value   -- root Q (mover's expected value, in [-1, 1])
            visit_counts -- dict action -> raw visit count
        """
        assert not state.is_chance(), "MCTS root must be a decision node (advance chance first)"
        root = Node(state)
        self._expand(root)
        if add_noise and self.dirichlet_alpha > 0.0:
            self._add_dirichlet_noise(root)

        for _ in range(self.num_simulations):
            self._simulate(root)

        visit_counts = {a: child.visit_count for a, child in root.children.items()}
        total = sum(visit_counts.values()) or 1
        visit_probs = {a: n / total for a, n in visit_counts.items()}
        return visit_probs, root.q_value, visit_counts

    # --- one simulation ---------------------------------------------------
    def _simulate(self, root: Node) -> None:
        node = root
        path = [node]
        while node.is_expanded and not node.state.is_terminal():
            if node.state.is_chance():
                action = self._sample_chance(node.state)
            else:
                action = self._puct_action(node)
            if action not in node.children:
                child_state = node.state.apply_action(action)
                child = Node(child_state, parent=node, prior=node.child_priors.get(action, 0.0))
                if child_state.is_chance():
                    child.is_expanded = True       # chance nodes need no network expansion
                node.children[action] = child
            node = node.children[action]
            path.append(node)

        leaf = node.state
        if leaf.is_terminal():
            leaf_player = leaf.current_player()
            value = leaf.returns()[leaf_player] / leaf.max_abs_return
        else:
            value = self._expand(node)             # network value, mover's perspective
            leaf_player = leaf.current_player()

        self._backup(path, value, leaf_player)

    # --- helpers ----------------------------------------------------------
    def _expand(self, node: Node) -> float:
        """Evaluate a decision leaf with the network; store priors; return value."""
        state = node.state
        policy, value = self.net.predict(state.encode(), state.legal_actions())
        node.child_priors = policy
        node.is_expanded = True
        return value

    def _puct_action(self, node: Node) -> int:
        sqrt_n = math.sqrt(node.visit_count)
        best_score, best_action = -float("inf"), -1
        for action, prior in node.child_priors.items():
            child = node.children.get(action)
            if child is not None and child.visit_count > 0:
                # child's Q is from the child's perspective; flip if it's the opponent's.
                same = child.state.current_player() == node.state.current_player()
                q_parent = child.q_value if same else -child.q_value
                n = child.visit_count
            else:
                q_parent, n = 0.0, 0
            score = q_parent + self.c_puct * prior * sqrt_n / (1 + n)
            if score > best_score:
                best_score, best_action = score, action
        return best_action

    def _sample_chance(self, state: GameState) -> int:
        actions, probs = zip(*state.chance_outcomes())
        return int(self.rng.choice(actions, p=probs))

    def _backup(self, path: list[Node], value: float, leaf_player: int) -> None:
        for node in path:
            node.visit_count += 1
            # value is expressed from leaf_player's perspective; flip for the other player.
            node.value_sum += value if node.state.current_player() == leaf_player else -value

    def _add_dirichlet_noise(self, root: Node) -> None:
        actions = list(root.child_priors.keys())
        if not actions:
            return
        noise = self.rng.dirichlet([self.dirichlet_alpha] * len(actions))
        for a, eta in zip(actions, noise):
            root.child_priors[a] = (1 - self.noise_eps) * root.child_priors[a] + self.noise_eps * eta


# --- action selection from a visit distribution ---------------------------
def select_action(visit_counts: dict[int, int], temperature: float = 1.0, rng=None) -> int:
    """Pick an action from visit counts.

    ``temperature == 0`` -> greedy (most-visited). Otherwise sample from
    ``visits ** (1/temperature)`` -- high temperature explores, low exploits.
    """
    rng = rng or np.random.default_rng()
    actions = list(visit_counts.keys())
    counts = np.array([visit_counts[a] for a in actions], dtype=np.float64)
    if temperature <= 1e-6:
        return int(actions[int(counts.argmax())])
    weights = counts ** (1.0 / temperature)
    weights /= weights.sum()
    return int(rng.choice(actions, p=weights))


def advance_through_chance(state: GameState, rng: np.random.Generator) -> GameState:
    """Sample dice until a decision or terminal node (the *real* roll in self-play)."""
    while state.is_chance():
        actions, probs = zip(*state.chance_outcomes())
        state = state.apply_action(int(rng.choice(actions, p=probs)))
    return state

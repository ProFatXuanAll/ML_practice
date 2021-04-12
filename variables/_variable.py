import math
from queue import Queue, PriorityQueue
from typing import Any, List, Tuple


class Variable:
    r"""Variable class.

    Attributes
    ==========
    bp_graph: List[Tuple[Variable, float, float]]
        Gradient-flow graph for backward pass.
        Each flow consists of two input units and one output unit.
        Output unit will contain references for both input units.
        Each referenced input unit will contain first order partial derivatives
        of output with respect to input.
        Parital derivatives were determined previously per operations.
    priority: int
        Priority used to determine which variable get popped from priority
        queue during backward pass.
        Priority is initialized as follow:

        - If variable were created by hand, then priority is initialized to
          ``0`` (the lowest priority).
        - If variable were created during forward pass, then priority is set to
          the largest priority of the two operands plus ``1`` (which priority
          is larger then both operands).
    grad: float
        First order derivatives of output with respect to input.
    val: float
        Value of the variable.

    Parameters
    ==========
    val: float
        Value of the variable.
    """

    def __init__(self, val: float):
        self.priority: int = 0
        self.bp_graph: List[Tuple[Variable, float, float]] = []
        self.grad: float = 0.0
        self.val: float = float(val)

    def __float__(self) -> float:
        r"""Return ``self.val``."""
        return self.val

    def __int__(self) -> int:
        r"""Return ``int(self.val)``."""
        return int(self.val)

    def __repr__(self) -> str:
        r"""Format output on terminal."""
        return f'{self.__class__.__name__}({self.val})'

    def __str__(self) -> str:
        r"""Return ``str(self.val)``."""
        return str(self.val)

    def __eq__(self, r_op: Any) -> bool:
        r"""Equality operation ``self == r_op``."""
        if id(self) == id(r_op):
            return True
        elif isinstance(r_op, Variable):
            return self.val == r_op.val
        elif isinstance(r_op, (float, int)):
            return self.val == r_op
        return False

    def __req__(self, l_op: Any) -> bool:
        r"""Equality operation ``l_op == self``."""
        if isinstance(l_op, (float, int)):
            return l_op == self.val
        return False

    def __lt__(self, r_op: Any) -> bool:
        r"""Equality operation ``self < r_op``."""
        if isinstance(r_op, Variable):
            return self.val < r_op.val
        elif isinstance(r_op, (float, int)):
            return self.val < r_op
        return False

    def __rlt__(self, l_op: Any) -> bool:
        r"""Equality operation ``l_op < self``."""
        if isinstance(l_op, (float, int)):
            return self.val > l_op.val
        return False

    def __gt__(self, r_op: Any) -> bool:
        r"""Equality operation ``self > r_op``."""
        if isinstance(r_op, Variable):
            return self.val > r_op.val
        elif isinstance(r_op, (float, int)):
            return self.val > r_op
        return False

    def __rgt__(self, l_op: Any) -> bool:
        r"""Equality operation ``l_op > self``."""
        if isinstance(l_op, (float, int)):
            return self.val < l_op.val
        return False

    def __add__(self, r_op: Any):
        r"""Addition operation ``self + r_op``."""
        out = Variable(0.0)

        if isinstance(r_op, Variable):
            out.val = self.val + r_op.val
            out.bp_graph.append((self, 1.0))
            out.bp_graph.append((r_op, 1.0))
            out.priority = max(self.priority, r_op.priority) + 1
        elif isinstance(r_op, (float, int)):
            out.val = self.val + r_op
            out.bp_graph.append((self, 1.0))
            out.priority = self.priority + 1
        else:
            return NotImplemented
        return out

    def __radd__(self, l_op: Any):
        r"""Addition operation ``l_op + self``."""
        out = Variable(0.0)

        if isinstance(l_op, (float, int)):
            out.val = l_op + self.val
            out.bp_graph.append((self, 1.0))
            out.priority = self.priority + 1
            return out
        return NotImplemented

    def __mul__(self, r_op: Any):
        r"""Multiplication operation ``self * r_op``."""
        out = Variable(0.0)

        if isinstance(r_op, Variable):
            out.val = self.val * r_op.val
            out.bp_graph.append((self, r_op.val))
            out.bp_graph.append((r_op, self.val))
            out.priority = max(self.priority, r_op.priority) + 1
        elif isinstance(r_op, (float, int)):
            out.val = self.val * r_op
            out.bp_graph.append((self, r_op))
            out.priority = self.priority + 1
        else:
            return NotImplemented
        return out

    def __rmul__(self, l_op: Any):
        r"""Multiplication operation ``l_op * self``."""
        out = Variable(0.0)

        if isinstance(l_op, (float, int)):
            out.val = l_op * self.val
            out.bp_graph.append((self, l_op))
            out.priority = self.priority + 1
            return out
        return NotImplemented

    def __sub__(self, r_op: Any):
        r"""Subtraction operation ``self - r_op``."""
        out = Variable(0.0)

        if isinstance(r_op, Variable):
            out.val = self.val - r_op.val
            out.bp_graph.append((self, 1.0))
            out.bp_graph.append((r_op, -1.0))
            out.priority = max(self.priority, r_op.priority) + 1
        elif isinstance(r_op, (float, int)):
            out.val = self.val - r_op
            out.bp_graph.append((self, 1.0))
            out.priority = self.priority + 1
        else:
            return NotImplemented
        return out

    def __rsub__(self, l_op: Any):
        r"""Subtraction operation ``l_op - self``."""
        out = Variable(0.0)

        if isinstance(l_op, (float, int)):
            out.val = l_op - self.val
            out.bp_graph.append((self, -1.0))
            out.priority = self.priority + 1
            return out
        return NotImplemented

    def __truediv__(self, r_op: Any):
        r"""Division operation ``self / r_op``."""
        out = Variable(0.0)

        if isinstance(r_op, Variable):
            out.val = self.val / r_op.val
            out.bp_graph.append((self, 1 / r_op.val))
            out.bp_graph.append((r_op, -self.val * (r_op.val ** (-2))))
            out.priority = max(self.priority, r_op.priority) + 1
        elif isinstance(r_op, (float, int)):
            out.val = self.val / r_op
            out.bp_graph.append((self, 1 / r_op))
            out.priority = self.priority + 1
        else:
            return NotImplemented
        return out

    def __rtruediv__(self, l_op: Any):
        r"""Division operation ``l_op / self``."""
        out = Variable(0.0)

        if isinstance(l_op, (float, int)):
            out.val = l_op / self.val
            out.bp_graph.append((self, -l_op * (self.val ** (-2))))
            out.priority = self.priority + 1
            return out
        return NotImplemented

    def __pow__(self, r_op: Any):
        r"""Exponential operation ``self ** r_op``."""
        out = Variable(0.0)

        if isinstance(r_op, Variable):
            out.val = self.val ** r_op.val
            out.bp_graph.append((
                self,
                r_op.val * (self.val ** (r_op.val - 1)),
            ))
            out.bp_graph.append((
                r_op,
                math.log(self.val) * (self.val ** r_op.val),
            ))
            out.priority = max(self.priority, r_op.priority) + 1
        elif isinstance(r_op, (float, int)):
            out.val = self.val ** r_op
            out.bp_graph.append((self, r_op * (self.val ** (r_op - 1))))
            out.priority = self.priority + 1
        else:
            return NotImplemented
        return out

    def __rpow__(self, l_op: Any):
        r"""Exponential operation ``l_op ** self``."""
        out = Variable(0.0)

        if isinstance(l_op, (float, int)):
            out.val = l_op ** self.val
            out.bp_graph.append((self, math.log(l_op) * (l_op ** self.val)))
            out.priority = self.priority + 1
            return out
        return NotImplemented

    def backward_pass(self):
        r"""Backward pass algorithm."""
        q = PriorityQueue()
        is_put = {id(self): True}

        for nxt_var, nxt_grad in self.bp_graph:
            # Initial gradients are scaled by 1.0.
            nxt_var.grad += nxt_grad
            q.put((
                # Next variable's priority in queue.
                -nxt_var.priority,
                # Next variable's reference.
                nxt_var,
            ))
            is_put[id(nxt_var)] = True

        # Travel backward pass graph which order was based on forward priority.
        while not q.empty():
            _, cur_var = q.get()

            for nxt_var, nxt_grad in cur_var.bp_graph:
                # Each forward pass will contribute gradient.
                # The formula for backward pass gradient is:
                # backward gradient times local gradient.
                # Here backward gradient means `cur_var.grad`
                # and local gradient means `nxt_grad`.
                nxt_var.grad += cur_var.grad * nxt_grad

                # Do not put same node into queue twice.
                if id(nxt_var) not in is_put:
                    q.put((
                        # Next variable's priority in queue.
                        -nxt_var.priority,
                        # Next variable's reference.
                        nxt_var,
                    ))
                is_put[id(nxt_var)] = True

    def reset_bp_graph(self):
        r"""Reset backward pass graph and all priorities and gradients in the graph."""
        q = Queue()

        self.priority = 0
        self.grad = 0.0

        for nxt_var, _ in self.bp_graph:
            q.put(nxt_var)

        self.bp_graph.clear()

        # Travel backward pass graph which order was based on forward priority.
        while not q.empty():
            cur_var = q.get()
            cur_var.priority = 0
            cur_var.grad = 0.0

            for nxt_var, _ in cur_var.bp_graph:
                q.put(nxt_var)

            cur_var.bp_graph.clear()


def coerce(value: Any) -> Variable:
    r"""Coerce value into ``Variable``."""
    if isinstance(value, Variable):
        return value
    elif isinstance(value, (float, int)):
        return Variable(value)
    else:
        raise TypeError(f'{type(value)} is not compatible with ``Variable``.')


def copy(var: Variable):
    return Variable(var.val)

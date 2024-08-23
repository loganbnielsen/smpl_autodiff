from collections import defaultdict
from enum import Enum


def init_global_state():
    global EXPR_ID
    EXPR_ID = 0


class Operation(Enum):
    MULTIPLICATION = "*"
    ADDITION = "+"


class Expr:
    def __init__(self, val, lhs=None, rhs=None, op=None, track_grad=False):
        global EXPR_ID
        self.id = EXPR_ID
        EXPR_ID += 1

        self.lhs = lhs
        self.rhs = rhs
        self.op = op
        self.val = val
        self.track_grad = track_grad

    def backward(self):
        grads = defaultdict(lambda: Expr(0))
        self._backward_helper(Expr(1), grads)
        return grads

    def _backward_helper(self, seed, node_id_to_hyponode):
        if self.track_grad:
            node_id_to_hyponode[self.id] += seed
        if self.op is None:
            return

        if self.lhs is not None:
            self.lhs.__send_back(self.rhs, self.op, seed, node_id_to_hyponode)
        if self.rhs is not None:
            self.rhs.__send_back(self.lhs, self.op, seed, node_id_to_hyponode)

    def __send_back(self, other, op, seed, node_id_to_hyponode):
        if op == Operation.ADDITION:
            self._backward_helper(seed, node_id_to_hyponode)
        elif op == Operation.MULTIPLICATION:
            self._backward_helper(seed * other, node_id_to_hyponode)

    def __add__(self, rhs):
        return Expr(self.val + rhs.val, self, rhs, Operation.ADDITION, False)

    def __mul__(self, rhs):
        return Expr(self.val * rhs.val, self, rhs, Operation.MULTIPLICATION, False)

    def __str__(self) -> str:
        if self.op is None:
            return f"{self.val}"
        return f"{{ ({self.lhs}) {self.op.value} ({self.rhs}); val = {self.val};"


def main():
    init_global_state()

    x = 2
    y = 7
    expr_x = Expr(x, track_grad=True)
    expr_y = Expr(y, track_grad=True)
    expr_2 = Expr(2)

    # f(x,y) = (x + y + 2) * (y * y)
    # f(11,7) = (11) * (49)
    # 539
    final_node = (expr_x + expr_y + expr_2) * (expr_y * expr_y)
    print("before backprop")
    print(final_node)

    grads = final_node.backward()

    # f(x,y) = xy^2 + y^3 + 2y^2
    # del f / del x = y^2
    #               = 49 @ (11,7)
    # del f / del y = 2xy + 3y^2 + 4y
    #               = 28 + 147 + 28 @ (11,7)
    #               = 203
    grad_x = grads[expr_x.id]
    grads_grad_x = grad_x.backward()
    # del^2 f / del x^2 = 0
    grad_xx = grads_grad_x[expr_x.id]
    print("grad_xx", grad_xx)
    # del^2 f / del y del x = 2y
    #                       = 2 * 7 @ (11,7)
    #                       = 14
    grad_yx = grads_grad_x[expr_y.id]
    print("grad_yx", grad_yx)
    grad_y = grads[expr_y.id]
    grads_grad_y = grad_y.backward()
    # del^2 f / del y^2 = 2x + 6y + 4
    #                   = 4 + 42 + 4 @ (11,7)
    #                   = 50
    grad_yy = grads_grad_y[expr_y.id]
    print("grad_yy", grad_yy)
    # del^2 f / del x del y = 2y
    #                       = 2 * 7 @ (11,7)
    #                       = 14
    grad_xy = grads_grad_y[expr_x.id]
    print("grad_xy", grad_xy)


if __name__ == "__main__":
    main()

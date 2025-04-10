import sympy
from sympy import symbols, diff, simplify, solve
"""方程求解库
"""

def quadratic_bezier_curve(p_0, p_1, p_2):
    """二阶贝塞尔曲线表达式

    Args:
        p_0, p_1, P_2: 关键点坐标[x, y]
    """
    t = symbols('t')
    x_t = (1-t)*(1-t) * p_0[0] + 2*t*(1-t)*p_1[0]+t*t*p_2[0]
    y_t = (1-t)*(1-t) * p_0[1] + 2*t*(1-t)*p_1[1]+t*t*p_2[1]
    return x_t, y_t, t
    pass

def min_distance_quadratic_bezier_curve(x_t, y_t, t, x, y):
    """求解点到二阶贝塞尔曲线的最小距离

    Args:
        x_t: 关于x的参数方程
        y_t: 关于y的参数方程
        t: 参数t
        (x, y): 点的坐标
    """
    p = (x_t - x)**2 + (y_t - y) ** 2
    dp_dt = simplify(diff(p, t))
    sovle_t = solve(dp_dt, t)
    print(sovle_t)
    min_dis = simplify(p.subs(t, sovle_t[0]))
    
    print(simplify(min_dis))
    print(p.subs(t, 0))
    print(p.subs(t, 1))
    pass
if __name__ == "__main__":
    x_t, y_t, t = quadratic_bezier_curve(
        [29.8, 22.2],
        [29.8, 29.5],
        [29.8, 36.7]
    )
    
    min_distance_quadratic_bezier_curve(x_t, y_t, t, 32, 32)
    pass
from sympy import sympify, solve, Eq, Symbol, And, Or, oo, Rational, expand
from sympy.matrices import Matrix, eye, zeros, ones, diag
from sympy.plotting import plot_implicit, plot3d
import csv

# UTILS #

def parse_matrix(input=None, file=None, ignore_header=False, ignore_rownames=False, new_var_name='tau'):

    def parse_row(row, new_var_name=new_var_name):
        temp_row = [0] * len(row)

        n_vars = 0
        list_vars = []
        constant_sum = 0
        n_stars = 0

        for i in range(len(row)):
            expr = row[i]
            if expr == '?':
                n_vars += 1
                temp_row[i] = Symbol(f'{new_var_name}_{n_vars}')
                list_vars.append(temp_row[i])
            elif expr == '*':
                n_stars += 1
            else:
                temp_row[i] = sympify(expr)
                if temp_row[i].is_constant():
                    temp_row[i] = Rational('{}'.format(temp_row[i]))
                    constant_sum += temp_row[i]
                else:
                    list_vars.append(temp_row[i])

        for i in range(len(row)):
            if row[i] == '*':
                temp_row[i] = (1 - constant_sum - sum(list_vars)) / n_stars

        return temp_row


    if input is not None:
        if not isinstance(input[0], list) or len(input)==1:
            if len(input) == 1:
                input = input[0]
            return Matrix(parse_row(input))
        else:
            return Matrix([parse_row(input[i], new_var_name=f'{new_var_name}_{i+1}') for i in range(len(input))])

    elif file is not None:
        file_content = [row for row in csv.reader(open(file, newline='', encoding='utf-8-sig'))]

        if ignore_header:
            file_content = file_content[1:]
        if ignore_rownames:
            file_content = [f[1:] for f in file_content]

        return parse_matrix(input = file_content, ignore_header=ignore_header, ignore_rownames=ignore_rownames)

    else:
        return


def ineq_to_eq(ineq):
    return Eq(ineq.lhs, ineq.rhs)


# converts a rational function f/g relation_type \in {>,<,<=,>=,=} rhs to f - g * rhs relation_type 0
# relation_type is relation[0], rhs is relation[1]
def rational_to_ineq(components, relation):
    relation_type = relation[0]
    rhs = relation[1]

    lhs = components[0] - components[1] * rhs

    match relation_type:
        case '=':
            return Eq(lhs, 0)
        case '<':
            return lhs < 0
        case '>':
            return lhs > 0
        case '<=':
            return lhs <= 0
        case '>=':
            return lhs >= 0
        case _:
            raise Exception('Relation type must be one of : <, >, <=, >=, or =.')


# ALL THE MARKOV/CEA ANALYSES

def finite_horizon_reward(P, pi, R, discount, horizon):
    P_sum = zeros(P.shape[0])
    P_pow = eye(P.shape[0])

    for t in range(horizon+1):
        if t > 0:
            P_pow = P_pow * P
        P_sum += P_pow * (Rational('{}'.format(discount))**t)

    result = expand(pi.T * P_sum * R)

    return result[0]


# returns the 'numerator' and 'denominator'
def infinite_horizon_reward(P, pi, R, discount):
    M = eye(P.shape[0]) - Rational('{}'.format(discount)) * P

    adj_component = pi.T * M.adjugate() * R
    adj_component = expand(adj_component[0])

    det_component = M.det()

    return adj_component, det_component


# n_recurrent: the first n_recurrent states are the non-death states
def total_reward(P, pi, R, n_recurrent):
    P_, pi_, R_ = P[0:n_recurrent, 0:n_recurrent], pi[0:n_recurrent, :], R[0:n_recurrent, :]
    M = eye(P_.shape[0]) - P_

    adj_component = pi_.T * M.adjugate() * R_
    adj_component = expand(adj_component[0])

    det_component = M.det()

    return adj_component, det_component


def finite_horizon_reward_ineq(P, pi, R, discount, horizon, relation=['>=', 0]):
    return rational_to_ineq([finite_horizon_reward(P, pi, R, discount, horizon), 1], relation)


def infinite_horizon_reward_ineq(P, pi, R, discount, relation=['>=', 0]):
    components = infinite_horizon_reward(P, pi, R, discount)
    return rational_to_ineq(components, relation)


def total_reward_ineq(P, pi, R, n_recurrent, relation=['>=', 0]):
    components = total_reward(P, pi, R, n_recurrent)
    return rational_to_ineq(components, relation)


# calculates ICER of intervention B compared to intervention A
def infinite_horizon_icer(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount):
    components_cost_a = infinite_horizon_reward(P_a, pi_a, cost_a, discount)
    components_benefit_a = infinite_horizon_reward(P_a, pi_a, benefit_a, discount)
    components_cost_b = infinite_horizon_reward(P_b, pi_b, cost_b, discount)
    components_benefit_b = infinite_horizon_reward(P_b, pi_b, benefit_b, discount)

    # note mathematically, components_cost_i[1] == components_benefit_i[1], because the det is purely based on the prob matrix
    det_a = components_cost_a[1]
    det_b = components_cost_b[1]

    return components_cost_b[0] * det_a - components_cost_a[0] * det_b, components_benefit_b[0] * det_a - components_benefit_a[0] * det_b

# returns icer of B vs A
def finite_horizon_icer(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount, horizon):
    delta_cost = finite_horizon_reward(P_b, pi_b, cost_b, discount, horizon) - finite_horizon_reward(P_a, pi_a, cost_a, discount, horizon)
    delta_benefit = finite_horizon_reward(P_b, pi_b, benefit_b, discount, horizon) - finite_horizon_reward(P_a, pi_a, benefit_a, discount, horizon)

    return delta_cost, delta_benefit


def infinite_horizon_icer_ineq(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount, relation=['>=', 0]):
    components = infinite_horizon_icer(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount)
    return rational_to_ineq(components, relation)


def finite_horizon_icer_ineq(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount, horizon, relation=['>=', 0]):
    components = finite_horizon_icer(P_a, pi_a, cost_a, benefit_a, P_b, pi_b, cost_b, benefit_b, discount, horizon)
    return rational_to_ineq(components, relation)


# returns the inequality that total reward of B >= total reward of A
def finite_horizon_reward_paired(P_a, pi_a, reward_a, P_b, pi_b, reward_b, discount, horizon):
    total_a = finite_horizon_reward(P_a, pi_a, reward_a, discount, horizon)
    total_b = finite_horizon_reward(P_b, pi_b, reward_b, discount, horizon)

    return total_b - total_a >= 0


def infinite_horizon_reward_paired(P_a, pi_a, reward_a, P_b, pi_b, reward_b, discount, diff=0):
    components_a = infinite_horizon_reward(P_a, pi_a, reward_a, discount)
    components_b = infinite_horizon_reward(P_b, pi_b, reward_b, discount)

    components = [components_b[0] * components_a[1] - components_a[0] * components_b[1] , components_a[1] * components_b[1]]

    return rational_to_ineq(components, ['>=', diff])


# PLOTTING

# possible modes
# if nvar==2, plottype can either be region or boundary
# if nvar==3, plottype can either be facet or surface, must specify a z_var
# if plottype is facet, then must also provide facet_options
# if plottype is surface, we cannot draw Boolean expressions and such...! this is a limitation of matplotlib. so constraints wont appear.
# facet_options is a list containing [z_label, (value, color), (value, color)... etc]
def plot_simple(ineq, constraints=None, plottype='region', nvar=2, z_var=None, facet_options=None, **kwargs):

    def add_constraints(to_plot, constraints):
        if constraints is not None:
            return And(to_plot, *constraints)
        else:
            return to_plot


    if nvar==2:
        match plottype:
            case 'region':
                return plot_implicit(add_constraints(ineq, constraints), **kwargs)
            case 'boundary':
                eq = ineq_to_eq(ineq)
                return plot_implicit(add_constraints(eq, constraints), **kwargs)
            case _:
                raise Exception("With 2 variables, plottype can only be 'region' or 'boundary'.")

    elif nvar==3:
        if z_var is None:
            raise Exception("With 3 variables, must specify z_var.")

        match plottype:
            case 'facet':
                eq = ineq_to_eq(ineq)
                z_label = facet_options[0]
                plots = [ plot_implicit( add_constraints(eq.subs(z_var, this_z[0]), constraints),
                                        show=False, line_color=this_z[1], title=f'{z_label} = {this_z[0]}', **kwargs)
                        for this_z in facet_options[1:] ]
                return plots
            case 'boundary':
                eq = ineq_to_eq(ineq)
                polynomial_to_plot = solve(eq, z_var)[0]
                return plot3d(polynomial_to_plot, kwargs.get('x_var'), kwargs.get('y_var'), **kwargs)
            case _:
                raise Exception("With 3 variables, plottype must either be 'facet' or 'boundary'.")

    else:
        raise Exception('The nvar parameter can only be 2 or 3. For more complicated analysis, construct a CAD.')


# PREPARING IT FOR CAD CONSTRUCTION

# if a vector is passed, it will make the simplex constraint to add them to one
# if a matrix is passed, it will make the simplex constraints to make it rowwise stochastic
# it will ignore expressions that already 'fill in' the stochasticity
# e.g., [a, b, 1-a-b]. If we naively sum this, then we get TRUE, bc a+b+1-a-b always equals 1.
# but what we really want is that a+b==1.
# if for some reason you don't want this you can just do Eq(sum(row), 1) by yourself!
def make_stochastic_constraints(matrix):
    if matrix.shape[0] == 1 or matrix.shape[1] == 1:
        constant_sum = 0
        variables = set()
        for element in matrix:
            if element.is_constant():
                constant_sum += element
            else:
                variables.update(element.free_symbols)

        if constant_sum > 1:
            return False
        elif len(variables) == 0 and constant_sum == 1:
            return True
        elif len(variables) == 0 and constant_sum < 1:
            return False
        else:
            return Eq(sum(list(variables)), 1 - constant_sum)

    else:
        return [make_stochastic_constraints(matrix[row_index, :]) for row_index in range(matrix.shape[1])]



from sympy import Interval, Rational, FiniteSet, EmptySet, latex, oo, var, Set, Symbol, sympify
from anytree import NodeMixin, RenderTree, AnyNode, PostOrderIter
import copy

# THE FOUNDATIONAL CLASSES FOR A CAD

class CadRootNode(NodeMixin):
    def __init__(self):
        super(NodeMixin, self).__init__()
        self.var = None
        self.sample = 0
        self.condition = EmptySet
        self.parent = None
        self.ancestors_closed = []


class CadNode(NodeMixin):
    def __init__(self, var, condition, parent=None, children=None):
        super(NodeMixin, self).__init__()
        self.var = var
        self.condition = condition # either interval or finite set

        if parent:
            self.parent = parent
        if children:
            self.children = children

        self.ancestors_closed = [p for p in self.ancestors + (self,) if not isinstance(p, CadRootNode)][::-1]

        if isinstance(condition, Interval):
            self.node_type = "interval"
            if self.condition.start == -oo:
                self.sample = self.condition.end - 1
            elif self.condition.end == oo:
                self.sample = self.condition.start + 1
            else:
                self.sample = Rational(1,2) * (self.condition.start + self.condition.end)
        elif isinstance(condition, FiniteSet):
            self.node_type = "point"
            self.sample = self.condition.inf
        else:
            raise Exception('Condition must be given either as an Interval or a FiniteSet.')

        # propagate for sample points
        self.sample = self.sample.subs([(ancestor.var, ancestor.sample) for ancestor in self.ancestors if ancestor.var is not None])


    # graft_parent maintains the closed_ancestors property for self and for all of self's descendants!
    # you can of course also use .parent = as usual
    # but I did not want to override the setter method
    # this should be used for example when gluing trees together!
    # I call it grafting because it's like surgery
    def graft_parent(self, new_parent):
        self.parent = new_parent
        self.ancestors_closed = [p for p in self.ancestors + (self,) if not isinstance(p, CadRootNode)][::-1]

        for node in self.descendants:
            node.ancestors_closed = [p for p in node.ancestors + (node,) if not isinstance(p, CadRootNode)][::-1]


# UTILS

# Function to find all leaf nodes in a tree
def find_leaf_nodes(tree):
    return list(tree.leaves)

# removes nodes in paths that are all 0 -- important for absorbing case
def remove_all_zeros(root):
    for leaf in find_leaf_nodes(root):
        if all(anc.condition == FiniteSet(0) for anc in leaf.ancestors_closed):
            leaf.parent = None

    for node in PostOrderIter(root, maxlevel=root.height):
        if node.is_leaf:
            node.parent = None

    return root


def as_latex(self, var):
    # point
    if isinstance(self, FiniteSet):
        return f'{var} = {latex(self.inf)}'

    # open interval
    if self.start == -oo and self.end == oo:
        return f'-\infty < {var} < \infty'
    elif self.start == -oo:
        return f'{var} < {latex(self.end)}'
    elif self.end == oo:
        return f'{var} > {latex(self.start)}'
    else:
        return f'{latex(self.start)} < {var} < {latex(self.end)}'

Set.as_latex = as_latex


# if a defining_poly is supplied, it will print the value of it after each leaf
def cad_print(tree, print_sample=False, defining_poly=None):
    if defining_poly:
        leaves = find_leaf_nodes(tree)
        for leaf in leaves:
            eval_node = CadNode(var=Symbol('f^*'),
                                condition=FiniteSet(defining_poly.subs([(anc.var, anc.sample) for anc in leaf.ancestors_closed])),
                                parent=leaf)

    for pre, fill, node in RenderTree(tree):
        if isinstance(node, CadRootNode):
            print("root")
        else:
            treestr = f'{pre} ${node.condition.as_latex(node.var)}$'
            if print_sample:
                treestr += f', sample={node.sample}'
            print(treestr)

    if defining_poly:
        for leaf in find_leaf_nodes(tree):
            leaf.parent = None


def cad_latex(node, file_path=None):
    def recurse(node, depth=0):
        indent = '\t' * (depth-1)  # Create indentation
        if node.is_root:
            return "\n".join(recurse(child, depth+1) for child in node.children)
        elif node.is_leaf:
            return f"{indent}[{{${node.condition.as_latex(node.var)}$}}]"
        else:
            return f"{indent}[{{${node.condition.as_latex(node.var)}$}}\n" + "\n".join(recurse(child, depth+1) for child in node.children) + f"\n{indent}]"


    latex_str = "\\begin{forest}\n[{}, phantom\n"
    latex_str += recurse(node)
    latex_str += "\n]\n\\end{forest}"

    if file_path:
        with open(file_path, 'w') as file:
            file.write(latex_str)

    return latex_str


# given a function g_0 \sum_1^n x_i g_i, and the x_vars, it returns [g_0, g_1,...]
def make_g_functions(fcn, xvars):
    fcn = fcn.expand()
    g_functions = []

    # g_0 is the term with no x's
    g_functions.append( fcn.subs([(curr_x, 0) for curr_x in xvars]) )

    # get the rest
    for curr_x in xvars:
        g_functions.append( fcn.coeff(curr_x) )

    return g_functions



# FUNCTIONS TO MAKE SIMPLEX CADS, GLUE THEM TOGETHER, AND EXTEND THEM TO A SIMPLEX-EXTENSIBLE FUNCTION

# creates CAD for \sum_i variables[i] == k <= 1
# specify rhs
# if sub=True, it will be the inequality \sum <= k -- note this only affects the last cell
# if constructing a simplex cad, set absorbing=False to ensure that no simplex is fully zero, ie last node can't be zero!
# returns the root of the tree
def simplex_cad(variables, rhs=1, sub=False, absorbing=False):

    if rhs > 1:
        raise Exception("RHS must be <=1 for this function to work.")

    # the actual recursive workhorse
    def _simplex_cad(depth, parent):

        if depth > len(variables):
            return

        curr_var = variables[depth-1]

        if depth == len(variables):
            if sub:
                valid_interval = Interval(0, rhs - sum(variables[:depth-1]))
            else:
                valid_interval = FiniteSet(rhs - sum(variables[:depth-1]))
        else:
            valid_interval = Interval(0, rhs - sum(variables[:depth-1]))

        valid_interval = valid_interval.subs([(anc.var, anc.condition.inf) for anc in parent.ancestors_closed if anc.node_type=="point"])

        if valid_interval == EmptySet: # don't make anymore
            return
        if valid_interval.measure == 0: # its a degen interval ie point
            node0 = CadNode(var=curr_var, condition=valid_interval, parent=parent)
            _simplex_cad(depth + 1, parent=node0)
        else:
            node1 = CadNode(var=curr_var, condition=FiniteSet(valid_interval.start), parent=parent)
            _simplex_cad(depth + 1, parent=node1)

            node2 = CadNode(var=curr_var, condition=Interval.open(valid_interval.start, valid_interval.end), parent=parent)
            _simplex_cad(depth + 1, parent=node2)

            node3 = CadNode(var=curr_var, condition=FiniteSet(valid_interval.end), parent=parent)
            _simplex_cad(depth + 1, parent=node3)

    # start the CAD construction with a root node
    root = CadRootNode()
    _simplex_cad(1, root)

    # if absorbing, can't have row where all are zero!
    if absorbing:
        root = remove_all_zeros(root)

    return root


# given a list of simplex cads with disjoint variable sets, glues them together
# does it in order
# returns root of the first tree
def glue_simplex_cads(cads):

    # recursive workhorse
    def _glue_simplex_cads(counter=1):
        if counter >= len(cads):
            return

        curr_cad = cads[counter-1]

        leaf_nodes = find_leaf_nodes(root)  # Find leaf nodes of the current tree

        for leaf in leaf_nodes:
            for child in cads[counter].children:  # Iterate through children of the second tree's root, ie the cells themselves!
                # Create a deepcopy of each child to ensure unique nodes
                copied_child = copy.deepcopy(child)
                # Set the parent of the copied child to the current leaf node
                copied_child.graft_parent(leaf)

        return _glue_simplex_cads(counter+1)

    root = cads[0]
    _glue_simplex_cads(1)
    return root


# takes a list of list of variables (where each variables[i] has same length)
def ifr_simplex_cad(variables):
    variables_len = len(variables)
    variables_lens = [len(vars) for vars in variables]
    if not all(x == variables_lens[0] for x in variables_lens):
        raise Exception("The number of variables in each simplex must be the same for the IFR condition.")
    else:
        variables_sublen = variables_lens[0]

    # the actual recursive workhorse
    # depth_i is which simplex you're on, depth_j is which variable in that simplex you're on (like nested loop)
    # pretty much the same as simplex_cad fcn but slightly diff formulas and accounts for depth_i and depth_j
    def _ifr_simplex_cad(depth_i, depth_j, parent):

        if depth_j > variables_sublen:
            return _ifr_simplex_cad(depth_i+1, 1, parent)
        if depth_i > variables_len:
            return

        curr_var = variables[depth_i-1][depth_j-1]

        # this is really what's different: making alpha_ij depend on previous alphas
        if depth_j == variables_sublen:
            valid_interval = FiniteSet(1 - sum(variables[depth_i-1][:-1]))
        else:
            if depth_i == 1:
                ifr_upper_bound = 1 - sum(variables[depth_i-1][:depth_j-1])
            else:
                ifr_upper_bound = sum(variables[depth_i-2][:depth_j]) - sum(variables[depth_i-1][:depth_j-1])
            valid_interval = Interval(0, ifr_upper_bound)

        valid_interval = valid_interval.subs([(anc.var, anc.condition.inf) for anc in parent.ancestors_closed if anc.node_type=="point"])

        if valid_interval == EmptySet: # don't make anymore
            return
        if valid_interval.measure == 0: # its a degen interval ie point
            node0 = CadNode(var=curr_var, condition=valid_interval, parent=parent)
            _ifr_simplex_cad(depth_i, depth_j + 1, parent=node0)
        else:
            node1 = CadNode(var=curr_var, condition=FiniteSet(valid_interval.start), parent=parent)
            _ifr_simplex_cad(depth_i, depth_j + 1, parent=node1)

            node2 = CadNode(var=curr_var, condition=Interval.open(valid_interval.start, valid_interval.end), parent=parent)
            _ifr_simplex_cad(depth_i, depth_j + 1, parent=node2)

            node3 = CadNode(var=curr_var, condition=FiniteSet(valid_interval.end), parent=parent)
            _ifr_simplex_cad(depth_i, depth_j + 1, parent=node3)

    # start the CAD construction with a root node
    root = CadRootNode()
    _ifr_simplex_cad(1, 1, root)
    return root


# given the root of a simplex, extend the CAD to g_0 + \sum x_i g_i \geq 0
def simplex_extensible_cad(simplex, g0, g, x):

    def _simplex_extensible_cad(depth=1, parent=None):
        if depth > len(x):
            return

        curr_var = x[depth-1]

        this_gi_eval = g[depth-1].subs([(anc.var, anc.sample) for anc in parent.ancestors_closed])

        # no root, bc denom == 0
        if this_gi_eval == 0:
            # x = 0
            node0 = CadNode(var=curr_var, condition=FiniteSet(0), parent=parent)
            _simplex_extensible_cad(depth + 1, parent=node0)

            # x > 0
            node1 = CadNode(var=curr_var, condition=Interval.open(0, oo), parent=parent)
            _simplex_extensible_cad(depth + 1, parent=node1)
        else:
            # this is the root of the polynomial, not a tree root!
            poly_root = (- g0 - sum([x[i] * g[i] for i in range(0, depth-1)])) / g[depth-1]
            # propagate equality constraints
            poly_root = poly_root.subs([(anc.var, anc.condition.inf) for anc in parent.ancestors_closed if anc.node_type=="point"])
            poly_root_eval = poly_root.subs([(anc.var, anc.sample) for anc in parent.ancestors_closed])

            if poly_root_eval <= 0:
                # x = 0
                node0 = CadNode(var=curr_var, condition=FiniteSet(0), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node0)

                # x > 0
                node1 = CadNode(var=curr_var, condition=Interval.open(0, oo), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node1)

            else:
                # x = 0
                node0 = CadNode(var=curr_var, condition=FiniteSet(0), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node0)

                # 0 < x < root
                node1 = CadNode(var=curr_var, condition=Interval.open(0, poly_root), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node1)

                # x = root
                node2 = CadNode(var=curr_var, condition=FiniteSet(poly_root), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node2)

                # x > root
                node3 = CadNode(var=curr_var, condition=Interval.open(poly_root, oo), parent=parent)
                _simplex_extensible_cad(depth + 1, parent=node3)


    # helps to sympify the g's in case some are constants
    g = [sympify(g_curr) for g_curr in g]

    # if simplex is none, i.e, empty, start new root
    if simplex is None:
        simplex = CadRootNode()

    # if simplex is empty, i.e., just a root
    if simplex.height == 0:
        _simplex_extensible_cad(depth=1, parent=simplex)
    # otherwise go thru leaves
    else:
        for leaf in find_leaf_nodes(simplex):
            _simplex_extensible_cad(depth=1, parent=leaf)

    # now prune the nodes based on sample point evaluation
    defining_poly = g0 + sum(x[i] * g[i] for i in range(len(x)))
    for leaf in find_leaf_nodes(simplex):
        sample_eval = defining_poly.subs([(anc.var, anc.sample) for anc in leaf.ancestors_closed])
        if sample_eval < 0:
            leaf.parent = None

    for node in PostOrderIter(simplex, maxlevel=simplex.height):
        if node.is_leaf:
            node.parent = None

    return simplex














# GRAVEYARD

'''
This was my failed attempt to resolve bounds when we have bounds on the variables in a simplex.
My lemma on Fourier-Motzkin definitely applies, but resolving this into a CAD is actually hard.
Check the example $a+b+c = 1, a \in [0,0.6], b \in [0,0.7], c \in [0,0.6]$.
Look at the shadow of this onto the x-y plane, and you will see that you need to split at some other points.
But, it is unclear how to get these splits. Unless you just run a full CAD...!


# to get the right interval, we have to intersect it with the bounds on curr_var
# however, reasoning about this is hard in general, e.g., say we know c = 1-a-b and c \in [0,1]
# observe that the minimium possible value of c assigns the maximums to the variables that came before it, and v.v.
# but which values to assign? bc even if b has bounds [0,1], it's real bounds are controlled by other variables...
# so instead take the condition!
if depth < len(variables):
    min_possible = valid_interval.start.subs([(anc.var, anc.condition.sup) for anc in ancestors_closed])
    max_possible = valid_interval.end.subs([(anc.var, anc.condition.inf) for anc in ancestors_closed])

    # cases to resolve the intervals
    # note I can't just take intersection, because the resolution depends on min/max_possible but I want to resolve valid_interval

    # valid_interval fully within curr_bound
    if min_possible > curr_bound[0] and max_possible < curr_bound[1]:
        valid_interval = valid_interval
    elif min_possible > curr_bound[0] and max_possible >= curr_bound[1]:
        valid_interval = Interval(valid_interval.start, curr_bound[1])
    elif min_possible <= curr_bound[0] and max_possible < curr_bound[1]:
        valid_interval = Interval(curr_bound[0], valid_interval.end)
    elif min_possible <= curr_bound[0] and max_possible >= curr_bound[1]:
        valid_interval = Interval(curr_bound[0], curr_bound[1])
    elif min_possible == curr_bound[0] or max_possible == curr_bound[0]:
        valid_interval = FiniteSet(curr_bound[0])
    elif min_possible == curr_bound[1] or max_possible == curr_bound[1]:
        valid_interval = FiniteSet(curr_bound[1])
    else: # no intersection
        return

elif depth == len(variables): # the last variable
    min_possible, max_possible = valid_interval.inf, valid_interval.inf

    min_possible = min_possible.subs([(anc.var, anc.condition.sup) for anc in ancestors_closed])
    max_possible = max_possible.subs([(anc.var, anc.condition.inf) for anc in ancestors_closed])

    # invalid
    # IS THIS CORRECT? giving weird answer for rhs=2...
    #if max_possible > curr_bound[1] or min_possible < curr_bound[0]:
    #    return
'''
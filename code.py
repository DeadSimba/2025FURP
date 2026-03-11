# add in Symbolic_KANLayer
def symbolic_formula(self, floating_digit=4):
    from sympy import symbols
    x_sym = symbols([f'x{i}' for i in range(self.in_dim)])
    output_exprs = []
    for j in range(self.out_dim):
        expr = 0
        for i in range(self.in_dim):
            if self.mask[j, i] > 0:  # only valid connections are processed
                a, b, c, d = self.affine[j, i].detach().numpy()
                f_sympy = self.funs_sympy[j][i]
                expr += c * f_sympy(a * x_sym[i] + b) + d
        output_exprs.append(ex_round(expr, floating_digit))
return output_exprs

# add in KAN class
def auto_symbolic(self, lib):
    for layer in range(len(self.kan_layers)):
        for j in range(self.width[layer+1]):
            for i in range(self.width[layer]):
                x = self.get_activations(layer, i)  # get the activation value
                y = self.get_post_activations(layer+1, j)
                best_r2 = -1
                best_fun = None
                for fun_name in lib:
                    _, r2 = self.kan_layers[layer].fit_symbolic(i, j, fun_name, x, y)
                    if r2 > best_r2:
                        best_r2 = r2
                        best_fun = fun_name
                self.kan_layers[layer].fix_symbolic(i, j, best_fun)

# call functions
lib = ['x','x^2','exp','log','sqrt','sin']  # candidate libraries
model.auto_symbolic(lib=lib)  # automatically select the best sign function
formula = model.symbolic_formula(floating_digit=4)[0]
print(formula) 
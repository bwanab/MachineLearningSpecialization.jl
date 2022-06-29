### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 0075a170-c83d-47ce-bea3-20f36f90a438
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ cf2721be-f563-11ec-28f9-61dc126e486b
using Plots, LinearAlgebra, Statistics

# ╔═╡ c0aeda70-9277-43c7-86c8-50a96bbb96e7
begin
	X_vals = [0.0, 1, 2, 3, 4, 5]
	X_train = reshape(X_vals, length(X_vals), 1)
	y_train = [0,0,0,1,1,1]
end

# ╔═╡ cd440c77-0a8f-411f-86ae-38f3f6348b5e
scatter(X_vals, y_train)

# ╔═╡ 892687a5-afbb-4bc5-bc75-617a5a7c8fb2
logistic_loss(y, f_wb_i) = y == 0 ? -log(1 - f_wb_i) : -log(f_wb_i)

# ╔═╡ a241be32-5312-4a94-9cf3-4609b9a518a8
begin
	f_wb_is = range(0.01, 1, 11)
	y1 = [logistic_loss(1, f_wb_i) for f_wb_i in f_wb_is]
	y0 = [logistic_loss(0, f_wb_i) for f_wb_i in f_wb_is]
	p1 = plot(f_wb_is, y1, title="y = 1", xaxis="fw,b(x)", yaxis="loss")
	p0 = plot(f_wb_is, y0, title="y = 0", xaxis="fw,b(x)", yaxis="loss")
	plot(p1, p0, size=(700, 350))
end

# ╔═╡ 60249571-c1f0-4c5e-8a19-f38247956c53
sigmoid(z) = 1.0 ./ (1.0 .+ exp.(-z))

# ╔═╡ 79b85ccb-fe85-4892-8b66-8598b8c30dbe
function logistic_cost(X, y, w, b)
	m = length(y)
	cost = 0
	for i in 1:m
		f_wb_i = sigmoid(w .* X[i,:] .+ b)
		log_loss(f_wb_i) = logistic_loss(y[i], f_wb_i)
		cost += sum(log_loss.(f_wb_i))
	end
	cost / m
end

# ╔═╡ e7335a54-4431-48c4-898e-e5fd3ac2cb73
logistic_cost(X_train, y_train, 1, 1)

# ╔═╡ 931c4cc6-7e1a-492e-b734-f822937c4f23
begin
	log_cost(w,b) = logistic_cost(X_train, y_train, w,b)
	ws = range(-6, 13, 50)
	bs = range(-20, 0, 50)
	cost_M = log_cost.(ws', bs)
end

# ╔═╡ 13507aef-eb2e-4d68-bd87-2c0cc1c7003b
surface(ws, bs, cost_M)

# ╔═╡ 5bdfa795-01ea-46ea-b389-f65fb9ead2d4
surface(ws, bs, log.(cost_M))

# ╔═╡ Cell order:
# ╠═cf2721be-f563-11ec-28f9-61dc126e486b
# ╠═0075a170-c83d-47ce-bea3-20f36f90a438
# ╠═c0aeda70-9277-43c7-86c8-50a96bbb96e7
# ╠═cd440c77-0a8f-411f-86ae-38f3f6348b5e
# ╠═892687a5-afbb-4bc5-bc75-617a5a7c8fb2
# ╠═a241be32-5312-4a94-9cf3-4609b9a518a8
# ╠═60249571-c1f0-4c5e-8a19-f38247956c53
# ╠═79b85ccb-fe85-4892-8b66-8598b8c30dbe
# ╠═e7335a54-4431-48c4-898e-e5fd3ac2cb73
# ╠═931c4cc6-7e1a-492e-b734-f822937c4f23
# ╠═13507aef-eb2e-4d68-bd87-2c0cc1c7003b
# ╠═5bdfa795-01ea-46ea-b389-f65fb9ead2d4

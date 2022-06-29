### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 1ce875ee-064c-4c9c-b303-d1b2c98e872a
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ 62f255b0-f581-11ec-1a6a-370e2f66ff25
using Statistics,Plots, LinearAlgebra

# ╔═╡ d39e03e3-dafb-4491-96c6-cc578f361c88
X_train = reduce(hcat, [[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])'

# ╔═╡ 0c14fe66-6771-4fc1-b59a-e67a7f11f627
y_train = [0,0,0,1,1,1]

# ╔═╡ a3029bf4-747b-4384-bece-8b5975419fe3
begin
	X = X_train'
	m1 = [x.I[2] for x in findall(1 .< prod(X, dims=1))]
	m2 = [x.I[2] for x in findall(1 .>= prod(X, dims=1))]
	scatter(X[1,m1], X[2,m1], xaxis=("x₁"), yaxis=("x₂"), markercolor=:red, label="y=1")
	scatter!(X[1,m2], X[2,m2], xaxis=("x₁"), yaxis=("x₂"), markercolor=:blue, label="y=0" )
end


# ╔═╡ 3fd9de77-fe9b-45d8-b988-2c1f6fac5c97
begin
	w_tmp = reshape([1,1], 1, 2)
	b_tmp = -3
	compute_cost_logistic(X_train, y_train, w_tmp, b_tmp)
end

# ╔═╡ 8ef83806-e874-4c16-b886-c24ee412ef29
begin
	x0 = 0:6
	x1 = 3 .- x0
	x1_other = 4 .- x0
	plot(x0, x1, lims=(0, 4))
	plot!(x0, x1_other)
	scatter!(X[1,m1], X[2,m1], xaxis=("x₁"), yaxis=("x₂"), markercolor=:red, label="y=1")
	scatter!(X[1,m2], X[2,m2], xaxis=("x₁"), yaxis=("x₂"), markercolor=:blue, label="y=0" )	
end

# ╔═╡ 4fea4ffe-a916-4371-b3c0-9403468769f7
begin
	w1 = reshape([1,1],2,1)
	w2 = reshape([1,1],2,1)
	b1 = -3
	b2 = -4
	println("cost for b = -3 :", compute_cost_logistic(X_train, y_train, w1, b1))
	println("cost for b = -4 :", compute_cost_logistic(X_train, y_train, w2, b2))
	
end

# ╔═╡ Cell order:
# ╠═62f255b0-f581-11ec-1a6a-370e2f66ff25
# ╠═1ce875ee-064c-4c9c-b303-d1b2c98e872a
# ╠═d39e03e3-dafb-4491-96c6-cc578f361c88
# ╠═0c14fe66-6771-4fc1-b59a-e67a7f11f627
# ╠═a3029bf4-747b-4384-bece-8b5975419fe3
# ╠═3fd9de77-fe9b-45d8-b988-2c1f6fac5c97
# ╠═8ef83806-e874-4c16-b886-c24ee412ef29
# ╠═4fea4ffe-a916-4371-b3c0-9403468769f7

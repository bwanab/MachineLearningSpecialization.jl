### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 1e41ecbe-f589-11ec-2dbd-db59718a30b4
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ aab5395c-e65a-4369-99d6-b353c4491623
using Statistics,Plots, LinearAlgebra

# ╔═╡ 6ac9bc0a-ae4b-4bad-9803-a25cbff2714b
X_train = reduce(hcat, [[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])'

# ╔═╡ 19743524-4fed-43c7-9374-02c0253de1f9
y_train = [0,0,0,1,1,1]

# ╔═╡ 3a1ef179-8374-44d7-8d68-a3d9ad3f2408
begin
	X = X_train'
	m1 = [x.I[2] for x in findall(1 .< prod(X, dims=1))]
	m2 = [x.I[2] for x in findall(1 .>= prod(X, dims=1))]
	scatter(X[1,m1], X[2,m1], xaxis=("x₁"), yaxis=("x₂"), markercolor=:red, label="y=1")
	scatter!(X[1,m2], X[2,m2], xaxis=("x₁"), yaxis=("x₂"), markercolor=:blue, label="y=0" )
end

# ╔═╡ 38b13cd6-82e7-421c-bd08-f7821763ca9f
begin
	w_tmp = [2.0, 3.0]
	b_tmp = 1.0
	dj_db_tmp, dj_dw_tmp = MachineLearningSpecialization.compute_gradient_logistic(X_train, y_train, w_tmp, b_tmp)
end

# ╔═╡ 96f632e1-789e-4511-a9e9-4bcdeb2c7179
begin
	w_in = zeros(length(X_train[1,:]))
	b_in = 0.0
	w_out, b_out = gradient_descent(X_train, y_train, w_in, b_in, compute_cost_logistic, compute_gradient_logistic, 0.1, 10000)
end

# ╔═╡ ec1bec91-5988-42be-a263-5ed4a8d82b55
X_train

# ╔═╡ 07223b06-1bf9-448f-87f3-7a3a00724eba
begin
	scatter(X[1,m1], X[2,m1], xaxis=("x₁"), yaxis=("x₂"), markercolor=:red, label="y=1")
	scatter!(X[1,m2], X[2,m2], xaxis=("x₁"), yaxis=("x₂"), markercolor=:blue, label="y=0" )
	x0 = -b_out/w_out[1]
	x1 = -b_out/w_out[2]
	plot!([0, x0], [x1, 0], fill=(3, 0.2, :blue))
end

# ╔═╡ Cell order:
# ╠═1e41ecbe-f589-11ec-2dbd-db59718a30b4
# ╠═aab5395c-e65a-4369-99d6-b353c4491623
# ╠═6ac9bc0a-ae4b-4bad-9803-a25cbff2714b
# ╠═19743524-4fed-43c7-9374-02c0253de1f9
# ╠═3a1ef179-8374-44d7-8d68-a3d9ad3f2408
# ╠═38b13cd6-82e7-421c-bd08-f7821763ca9f
# ╠═96f632e1-789e-4511-a9e9-4bcdeb2c7179
# ╠═ec1bec91-5988-42be-a263-5ed4a8d82b55
# ╠═07223b06-1bf9-448f-87f3-7a3a00724eba

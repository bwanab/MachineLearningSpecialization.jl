### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 6dc368cc-f64b-11ec-0d6d-d300a8e0f6d1
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ 2b381356-c0e6-4f03-a0df-a23a332b2bfd
using Random

# ╔═╡ 6eb66146-44c4-446a-9bd7-3beed88df934
begin
	Random.seed!(1)
	X_tmp = rand(5,6)
	y_tmp = [0,1,0,1,0]
	w_tmp = rand(6) .- 0.5
	b_tmp = 0.5
	λ_tmp = 1
	compute_cost(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp)
end

# ╔═╡ fe3e9829-1bca-4fd6-af78-461d4a8f2636
c = compute_cost_logistic(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp)

# ╔═╡ 39b3a3f9-fff5-48d8-aa84-99b8f334487c
begin
	X_tmp1 = rand(5,3)
	w_tmp1 = rand(3)
	b, w = compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp)
end

# ╔═╡ a8f634c7-d2e6-4f19-a062-f6fe78db074c
compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp, logistic=true)

# ╔═╡ e5f3be8d-b923-4ce9-8157-d8c69354eafb
sum(w)

# ╔═╡ Cell order:
# ╠═2b381356-c0e6-4f03-a0df-a23a332b2bfd
# ╠═6dc368cc-f64b-11ec-0d6d-d300a8e0f6d1
# ╠═6eb66146-44c4-446a-9bd7-3beed88df934
# ╠═fe3e9829-1bca-4fd6-af78-461d4a8f2636
# ╠═39b3a3f9-fff5-48d8-aa84-99b8f334487c
# ╠═a8f634c7-d2e6-4f19-a062-f6fe78db074c
# ╠═e5f3be8d-b923-4ce9-8157-d8c69354eafb

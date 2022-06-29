### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ 0aaff635-cde3-4849-bd49-6b3578a0dc9b
begin
	import Pkg
	Pkg.develop(path=homedir()*"/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ 700b9f9e-f495-11ec-34a6-a580f98712ae
using Statistics, Plots, LinearAlgebra, Printf

# ╔═╡ 8a4a1c43-e76e-4929-9164-8c915deadff6
exp.([1,2,3])

# ╔═╡ 8c94a9ab-c449-4027-bf21-bd723925f87b
exp(1)

# ╔═╡ a2febc54-bbd2-4051-a91f-63f492a0863c
sigmoid(z::AbstractVector) = 1 ./ (1 .+ exp.(-z))

# ╔═╡ 0f0a16fd-1f9c-4930-a944-4df42ba0420e
z_tmp = range(-10, 10)

# ╔═╡ d5477486-4ae8-40c4-8f0b-4199b7e76f55
y = sigmoid(z_tmp);

# ╔═╡ 6cfe7f75-29c4-4d99-9169-7aab309a1328
for (z, y) in zip(z_tmp, y)
	@printf("% .3e %.3e\n", z, y)
end

# ╔═╡ 10144623-64c8-4d46-b525-8888def4ad40
begin
	plot(z_tmp, y, xaxis=("z"), yaxis=("sigmoid(z)"), title="Sigmoid Function")
	hline!([0.5])
	vline!([0.0])
end

# ╔═╡ 6d02b945-047b-4fab-b2cb-43d33411327b
md"## Logistic Regression"

# ╔═╡ b5b96e92-ab78-44f2-a31a-17d6c3d5762f
begin
	x_train = reshape([0., 1, 2, 3, 4, 5, 10], 7, 1)
	y_vals = [0, 0, 0, 1,1,1,1]
	y_train = y_vals .* 200 .- 100
end

# ╔═╡ 31cccf8e-52e0-4b5c-8a9a-c817a3507f96
w, b = run_gradient_descent(x_train, y_train, 10000, 1e-2)

# ╔═╡ bdedb76b-de81-463e-b69c-c9553a5ec9f7
z = reshape(x_train * w .+ b, size(x_train)[1])

# ╔═╡ 04e49a9a-afc2-4899-888d-f514671720ca
s = sigmoid(z)

# ╔═╡ bf2e0502-b5e5-4350-bae3-00bde3c7b2fd
plot(x_train, s)

# ╔═╡ Cell order:
# ╠═0aaff635-cde3-4849-bd49-6b3578a0dc9b
# ╠═700b9f9e-f495-11ec-34a6-a580f98712ae
# ╠═8a4a1c43-e76e-4929-9164-8c915deadff6
# ╠═8c94a9ab-c449-4027-bf21-bd723925f87b
# ╠═a2febc54-bbd2-4051-a91f-63f492a0863c
# ╠═0f0a16fd-1f9c-4930-a944-4df42ba0420e
# ╠═d5477486-4ae8-40c4-8f0b-4199b7e76f55
# ╠═6cfe7f75-29c4-4d99-9169-7aab309a1328
# ╠═10144623-64c8-4d46-b525-8888def4ad40
# ╟─6d02b945-047b-4fab-b2cb-43d33411327b
# ╠═b5b96e92-ab78-44f2-a31a-17d6c3d5762f
# ╠═31cccf8e-52e0-4b5c-8a9a-c817a3507f96
# ╠═bdedb76b-de81-463e-b69c-c9553a5ec9f7
# ╠═04e49a9a-afc2-4899-888d-f514671720ca
# ╠═bf2e0502-b5e5-4350-bae3-00bde3c7b2fd

### A Pluto.jl notebook ###
# v0.19.4

using Markdown
using InteractiveUtils

# ╔═╡ a4cda8b6-f289-11ec-0231-cb155b80b475
begin
	import Pkg
	Pkg.develop(path="/Users/williamallen/src/MachineLearningSpecialization")
	using MachineLearningSpecialization
end


# ╔═╡ 948a38b5-ebd6-4340-b771-cdb15a05b5a8
using LinearAlgebra, Statistics, PlutoUI,Plots,Printf

# ╔═╡ 68d9dba9-b2fa-47f0-b6eb-ceab31ab7d1c
TableOfContents()

# ╔═╡ ed73175e-5013-41d5-a77d-72436607ead3
md"## Polynomial Features"

# ╔═╡ 7dcd13b5-1172-4a5f-b340-686906ac9435
begin
	x = range(0, 19)
	y = 1 .+ x .^ 2
	X = reshape(x, length(x), 1)
end

# ╔═╡ cdcfbd77-c27a-4f0c-b01d-444471482873
model_w, model_b, _ = run_gradient_descent(X, y,1000, 1e-2);

# ╔═╡ f639da2f-a6a4-43f4-9334-d479e8e7c5e1
begin
	plot(x, y, label="Actual Value")
	scatter!(x, X * model_w, label="Predicted Value")
	plot!(title = "No Feature engineering", xaxis=("X"), yaxis=("Y") )
end

# ╔═╡ 09931dfe-fff8-4160-8e6e-19a9133ddbc9
begin
	X1 = reshape(x.^2, length(x), 1) # engineered feature
	model_w1, model_b1,_ = run_gradient_descent(X1, y, 10000, 1e-5)
end;

# ╔═╡ 28a0c91a-7034-401a-8a9a-bb04acddb53e
begin
	plot(x, y, label="Actual Value")
	scatter!(x, X1 * model_w1 .+ model_b1, label="Predicted Value")
	plot!(title = "Added x ^ 2 feature", xaxis=("X"), yaxis=("Y") )
end

# ╔═╡ 8cae4ed3-403b-40c7-8448-c19555060899
md"## Selecting Features"

# ╔═╡ d0b80583-1fee-48b1-a194-631ab7507b58
X2 = reduce(hcat, [x, x.^2, x.^3]);

# ╔═╡ 7d8c327b-545d-4a49-a782-3435bead933c
model_w2, model_b2,_ = run_gradient_descent(X2, y, 10000, 1e-7);

# ╔═╡ 0790f345-31d5-41c0-b9f4-cbf6c4cd6a97
model_w2, model_b2

# ╔═╡ 63be9c88-9c94-4406-8914-f8de8d3b8a37
begin
	plot(x, y, label="Actual Value")
	scatter!(x, X2 * model_w2 .+ model_b2, label="Predicted Value")
	plot!(title = "Added x^2, x^3 feature", xaxis=("X"), yaxis=("Y") )
end

# ╔═╡ f33fbf97-99e1-4754-9a58-f28d2cdfe6d6
md"## An Alternate View"

# ╔═╡ 627ae122-9811-4cd6-9583-11b5096744ee
begin
	p1 = scatter(X2[:, 1], y, xaxis=("x"))
	p2 = scatter(X2[:, 2], y, xaxis=("x^2"))
	p3 = scatter(X2[:, 3], y, xaxis=("x^3"))
	plot(p1, p2, p3, yaxis=("y"), legend=false)
end

# ╔═╡ 99c7f34d-48ef-4a38-936c-9c1cca601cc2
md"## Scaling Features"

# ╔═╡ 8432391a-f575-45eb-97b9-a1bc0f467f7f
X3,_,_ = MachineLearningSpecialization.zscore_normalize_features(X2);

# ╔═╡ 285264fd-dae2-4447-b67a-0f2ada037197
model_w3, model_b3 = run_gradient_descent(X3, y, 100000, 1e-1);

# ╔═╡ 42e1d13a-062f-4c8a-aa12-d30c17c6c432
model_w3, model_b3

# ╔═╡ 8c2bdd47-ebe0-4b98-9a7a-5d9c26e1a0b0
begin
	plot(x, y, label="Actual Value")
	scatter!(x, X3 * model_w3 .+ model_b3, label="Predicted Value")
	plot!(title = "Normalized  x, x^2, x^3 feature", xaxis=("X"), yaxis=("Y") )
end

# ╔═╡ 9c6feaaf-f5da-4a04-9512-f037be0bb43e
md"## Complex Functions"

# ╔═╡ 13374a55-3837-4d59-a17b-7fea39880495
y2 = cos.(x/2)

# ╔═╡ bff18139-e450-47df-b86e-9bd12076dc98
X4 = reduce(hcat, [x, x.^2, x.^3, x.^4, x.^5, x.^6, x.^7, x.^8, x.^9, x.^10, x.^11, x.^12, x.^13])

# ╔═╡ bb677ded-868f-4636-9e53-8af17cef410b
X4_norm,_,_ = MachineLearningSpecialization.zscore_normalize_features(X4);

# ╔═╡ e6873d0a-efdf-47cd-a1b8-f499b712ee82
model_w4, model_b4, hist = run_gradient_descent(X4_norm, y2, 1000000, 1e-1);

# ╔═╡ c585aba8-0df2-43cd-91d6-0f18ebcabbe2
model_w4

# ╔═╡ 9bef46ee-fa71-4fdf-b2ea-1b88ec4e7789
X4_norm * model_w4

# ╔═╡ 62c8ae1d-a004-44e1-b199-470516509730
begin
	plot(x, y2, label="Actual Value")
	scatter!(x, X4_norm * model_w4 .+ model_b4, label="Predicted Value")
	plot!(title = "Normalized  x, x^2, x^3...x^13 feature", xaxis=("X"), yaxis=("Y") )
end

# ╔═╡ Cell order:
# ╠═a4cda8b6-f289-11ec-0231-cb155b80b475
# ╠═948a38b5-ebd6-4340-b771-cdb15a05b5a8
# ╠═68d9dba9-b2fa-47f0-b6eb-ceab31ab7d1c
# ╟─ed73175e-5013-41d5-a77d-72436607ead3
# ╠═7dcd13b5-1172-4a5f-b340-686906ac9435
# ╠═cdcfbd77-c27a-4f0c-b01d-444471482873
# ╠═f639da2f-a6a4-43f4-9334-d479e8e7c5e1
# ╠═09931dfe-fff8-4160-8e6e-19a9133ddbc9
# ╠═28a0c91a-7034-401a-8a9a-bb04acddb53e
# ╟─8cae4ed3-403b-40c7-8448-c19555060899
# ╠═d0b80583-1fee-48b1-a194-631ab7507b58
# ╠═7d8c327b-545d-4a49-a782-3435bead933c
# ╠═0790f345-31d5-41c0-b9f4-cbf6c4cd6a97
# ╠═63be9c88-9c94-4406-8914-f8de8d3b8a37
# ╟─f33fbf97-99e1-4754-9a58-f28d2cdfe6d6
# ╠═627ae122-9811-4cd6-9583-11b5096744ee
# ╟─99c7f34d-48ef-4a38-936c-9c1cca601cc2
# ╠═8432391a-f575-45eb-97b9-a1bc0f467f7f
# ╠═285264fd-dae2-4447-b67a-0f2ada037197
# ╠═42e1d13a-062f-4c8a-aa12-d30c17c6c432
# ╠═8c2bdd47-ebe0-4b98-9a7a-5d9c26e1a0b0
# ╟─9c6feaaf-f5da-4a04-9512-f037be0bb43e
# ╠═13374a55-3837-4d59-a17b-7fea39880495
# ╠═bff18139-e450-47df-b86e-9bd12076dc98
# ╠═bb677ded-868f-4636-9e53-8af17cef410b
# ╠═e6873d0a-efdf-47cd-a1b8-f499b712ee82
# ╠═c585aba8-0df2-43cd-91d6-0f18ebcabbe2
# ╠═9bef46ee-fa71-4fdf-b2ea-1b88ec4e7789
# ╠═62c8ae1d-a004-44e1-b199-470516509730

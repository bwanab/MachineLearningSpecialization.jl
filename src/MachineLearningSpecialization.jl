module MachineLearningSpecialization

using LinearAlgebra, Statistics, Printf, DataFrames, Flux

export compute_cost, compute_gradient, gradient_descent, run_gradient_descent,
    zscore_normalize_features, sigmoid, logistic_loss, compute_cost_logistic,
    compute_gradient_logistic

function compute_cost(X, y, w, b; λ = 0)
	m = size(X)[1]
	function do_one(i)
		f_wb_i = X[i,:] ⋅ w + b
		(f_wb_i - y[i]) ^ 2
	end
	cost = sum(do_one(i) for i in 1:m) / (2 * m)
    reg_cost = sum(w .^ 2) * (λ / (2 * m))
    cost + reg_cost
end

logistic_loss(y, f_wb_i) = y == 0 ? -log(1 - f_wb_i) : -log(f_wb_i)

function compute_cost_logistic(X, y, w, b; λ = 0)
    m = size(X)[1]
	cost = 0
	for i in 1:m
		f_wb_i = sigmoid(w ⋅ X[i,:] + b)
        println("f_wb_i = ", f_wb_i)
		log_loss(f_wb_i) = logistic_loss(y[i], f_wb_i)
		cost += log_loss(f_wb_i)
	end
	cost = cost / m
    reg_cost = sum(w .^ 2) * (λ / (2 * m))
    cost + reg_cost
end

function compute_gradient(X, y, w, b; logistic=false, λ = 0)
	m,n = size(X)
	dj_dw = zeros(n)
	dj_db = 0.0
	for i in 1:m
        d = X[i,:] ⋅ w
		err = (logistic ? (sigmoid(d + b)) : (d + b)) - y[i]
		for j in 1:n
			dj_dw[j] = dj_dw[j] + err * X[i, j]
		end
		dj_db = dj_db + err
	end
	dj_db = dj_db / m
    dj_dw = dj_dw / m .+ (λ / m) .* w
    dj_db, dj_dw
end

compute_gradient_logistic(X, y, w, b) = compute_gradient(X, y, w, b, logistic=true)

function gradient_descent(X, y, w_in, b_in, cost_func, gradient_func, alpha, num_iters)
	J_history = []
	w = deepcopy(w_in)
	b = b_in
    save_interval = ceil(num_iters / 10000)
	for i in 0:num_iters-1
		dj_db, dj_dw = gradient_func(X, y, w, b)
		w = w - alpha * dj_dw
		b = b - alpha * dj_db
		if i == 0 || i % save_interval == 0
			push!(J_history, [i, cost_func(X, y, w, b), w, b, dj_dw, dj_db])
		end
	end
	w, b, J_history
end

function run_gradient_descent(X, y, iterations, alpha)
    num_features = size(X)[2]
    initial_w = zeros(num_features)
    initial_b = 0
    w_final, b_final, J_hist = gradient_descent(X, y, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
    col_strings = reduce(vcat, [["Iteration", "Cost"], ["w"*string(i) for i in 1:num_features], 
            ["b"], ["djdw"*string(i) for i in 1:num_features], ["djdb"]])
    cols = Symbol.(col_strings)
    function make_row(cols, ragged_vals, i)
        vals = collect(Iterators.flatten(ragged_vals))
        (;zip(cols, vals)...)
    end
    hist = DataFrame([make_row(cols, J_hist[i], i) for i in 1:length(J_hist)])
    w_final, b_final, hist
end

predict(X, w, b) = X * w .+ b


function zscore_normalize_features(X)
	mu = mean(X, dims=1)
	sigma = std(X, dims=1)
	X_norm = (X .- mu) ./ sigma
	X_norm, mu, sigma
end

sigmoid(z::AbstractVector) = 1.0 ./ (1.0 .+ exp.(-z))
sigmoid(z) = 1.0 / (1.0 + exp(-z))

struct Norm

end
using MachineLearningSpecialization, LinearAlgebra
using Test

@testset "MachineLearningSpecialization.jl" begin
    X_train = [2104 5 1 45;
                1416 3 2 40;
                852 2 1 35]
    y_train = [460, 232, 178]
    b_init = 785.1811367994083
    w_init = [ 0.39133535, 18.75376741, -53.36032453, -26.42131618]
    @test compute_cost(X_train, y_train, w_init, b_init) ≈ 1.5578904428966628e-12
    initial_w = zeros(length(w_init))
	initial_b = 0
	iterations = 1000
	alpha = 7.0e-7
	w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)
    predictions = [X_train[i,:] ⋅ w_final + b_final for i in 1:size(X_train)[1]]
    @test predictions ≈ [426.4158681489784, 286.04856570554927, 171.096097572005]
    x_vals = [[4.17022005e-01, 7.20324493e-01, 1.14374817e-04, 3.02332573e-01,
                1.46755891e-01, 9.23385948e-02],
               [1.86260211e-01, 3.45560727e-01, 3.96767474e-01, 5.38816734e-01,
                4.19194514e-01, 6.85219500e-01],
               [2.04452250e-01, 8.78117436e-01, 2.73875932e-02, 6.70467510e-01,
                4.17304802e-01, 5.58689828e-01],
               [1.40386939e-01, 1.98101489e-01, 8.00744569e-01, 9.68261576e-01,
                3.13424178e-01, 6.92322616e-01],
               [8.76389152e-01, 8.94606664e-01, 8.50442114e-02, 3.90547832e-02,
                1.69830420e-01, 8.78142503e-01]]
    X_tmp = reduce(hcat, x_vals)'
    y_tmp = [0,1,0,1,0]
    w_tmp = [-0.40165317, -0.07889237,  0.45788953,  0.03316528,  0.19187711,
               -0.18448437]
    b_tmp = 0.5
    λ_tmp = 0.7
    @test compute_cost_logistic(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp) ≈ 0.685084914328802
    @test compute_cost(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp) ≈ 0.0791723937669179
    x_vals = [[4.17022005e-01, 7.20324493e-01, 1.14374817e-04],
        [3.02332573e-01, 1.46755891e-01, 9.23385948e-02],
        [1.86260211e-01, 3.45560727e-01, 3.96767474e-01],
        [5.38816734e-01, 4.19194514e-01, 6.85219500e-01],
        [2.04452250e-01, 8.78117436e-01, 2.73875932e-02]]
    X_tmp = reduce(hcat, x_vals)'
    y_tmp = [0,1,0,1,0]
    w_tmp = [0.67046751, 0.4173048 , 0.55868983]
    b_tmp = 0.5
    λ_tmp = 0.7
    b, w = compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp)
    @test w ≈ [0.296532147198904, 0.4911679613448304, 0.21645877537941188]
    @test b ≈ 0.6648774559852634
    b, w = compute_gradient(X_tmp, y_tmp, w_tmp, b_tmp, λ = λ_tmp, logistic=true)
    @test w ≈ [0.17380012926644225, 0.32007507822698794, 0.10776313414551181]
    @test b ≈ 0.3417989947913509

    states = []
    push!(states, State(100, true))
    push!(states, State(0, false))
    push!(states, State(0, false))
    push!(states, State(0, false))
    push!(states, State(0, false))
    push!(states, State(40, true))
    model = Model(states, [:left, :right, :either], 0.5)

    @test bellman(model, 3) == 25
    @test bellman(model, 5) == 20

end

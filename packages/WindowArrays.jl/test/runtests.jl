using WindowArrays
using Test: @test, @test_throws, @testset
import Random

populate_fft_plan_cache(
  [:plan_fft, :plan_fft!, :plan_rfft, :plan_bfft!, :plan_brfft],
  [Float64, ComplexF64],
  2 .^ 1:10)
@testset "`WindowMatrix` multiplications" begin
  @testset "`WindowMatrix` multiplications for n=$(n); k=$(k); A=$(A); B=$(B); t=$(t); prepare=:$(prepare); ais=$(ais)" for
      n in [0, 1, 10, 100],
      k in [0, 1, 10],
      A in [Float64, ComplexF64],
      B in [Float64, ComplexF64],
      t in [Real, Complex],
      prepare in [:matvec, :adjvec],
      ais in [true, false]
    if k ≤ n || n == 0
      ai = collect(1 : 5 : n - k)
      if !ais
        Random.shuffle!(ai)
      end
      a = WindowMatrix(rand(A, n), ai, k)
      am = materialize(a)
      if A <: Complex && t == Real
        @test_throws AssertionError mul_prepare(a; type = t)
      else
        a = mul_prepare(a;
          matvec = prepare == :matvec, adjvec = prepare == :adjvec,
          plan_bp = true, plan_brp = true, plan_ip = true, plan_irp = true,
          matvec_minimal_payload_size = 10, type = t)
        b0 = rand(B, k)
        b1 = rand(B, length(ai))
        tests_atol = 1e-12
        @test !any(isnan.(a * b0))
        @test a * b0 ≈ am * b0 atol = tests_atol
        @test !any(isnan.(a' * b1))
        @test a' * b1 ≈ am' * b1 atol = tests_atol
        @test !any(isnan.(transpose(a) * b1))
        @test transpose(a) * b1 ≈ transpose(am) * b1 atol = tests_atol
        @test sum(a; dims = 1) ≈ sum(am; dims = 1) atol = tests_atol
        @test sum(a; dims = 2) ≈ sum(am; dims = 2) atol = tests_atol
        @test sum(a; dims = :) ≈ sum(am; dims = :) atol = tests_atol
        @test sum(a'; dims = 1) ≈ sum(am'; dims = 1) atol = tests_atol
        @test sum(a'; dims = 2) ≈ sum(am'; dims = 2) atol = tests_atol
        @test sum(a'; dims = :) ≈ sum(am'; dims = :) atol = tests_atol
        @test sum(transpose(a); dims = 1) ≈ sum(transpose(am); dims = 1) atol = tests_atol
        @test sum(transpose(a); dims = 2) ≈ sum(transpose(am); dims = 2) atol = tests_atol
        @test sum(transpose(a); dims = :) ≈ sum(transpose(am); dims = :) atol = tests_atol
      end
    end
  end
end

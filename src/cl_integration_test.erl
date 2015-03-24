-module(cl_integration_test).


-include_lib("eunit/include/eunit.hrl").

-include_lib("emas/include/emas.hrl").

-compile(export_all).

run_test() ->
  SimParams = #sim_params{ problem_size = 120},
  Solution = labs_ops:solution(SimParams),
  Energy = labs_ops:energy(Solution),
  io:format(user, ">>> Energy: ~p~n", [Energy]).



same_evaluation_test() ->
  same_evaluation_test(40).


same_evaluation_test(Size) ->
  SimParams = #sim_params{problem_size = Size},
  Solution = labs_ops:solution(SimParams),
  Fitnes = labs_ops:evaluation(Solution, SimParams),

  BinarySolution = erlang:list_to_binary(Solution),
  BinaryFitness = labs_bin_ops:evaluation(BinarySolution, SimParams),

  Diff = Fitnes - BinaryFitness,

  io:format(">>> Fitness: ~p~n"
            ">>> BinFitn: ~p~n", [Fitnes, BinaryFitness]),
  ?assert(-0.000001 < Diff andalso Diff < 0.000001).



same_energy_test() ->
  same_energy_test(10).

same_energy_test(Size) ->
  SimParams = #sim_params{problem_size = Size},
  Solution = labs_ops:solution(SimParams),
  Energy = labs_ops:energy(Solution),


  BinarySolution = erlang:list_to_binary(Solution),
  BianryEnergy = labs_bin_ops:energy(BinarySolution),

  ?assertEqual(Energy,
               BianryEnergy),
  io:format(user, ">>> Energy: ~p~n", [Energy]).



test_avg(M, F, A, N) when N > 0 ->
    L = test_loop(M, F, A, N, []),
    Length = length(L),
    Min = lists:min(L),
    Max = lists:max(L),
    Med = lists:nth(round((Length / 2)), lists:sort(L)),
    Avg = round(lists:foldl(fun(X, Sum) -> X + Sum end, 0, L) / Length),
    io:format("Range: ~b - ~b mics~n"
              "Median: ~b mics~n"
              "Average: ~b mics~n",
              [Min, Max, Med, Avg]),
    Med.

test_loop(_M, _F, _A, 0, List) ->
    List;
test_loop(M, F, A, N, List) ->
    {T, _Result} = timer:tc(M, F, A),
    test_loop(M, F, A, N - 1, [T|List]).


time_it_test() ->
  time_it_test(40).

time_it_test(Size) ->
  SimParams = #sim_params{problem_size = Size},
  Solution = labs_ops:solution(SimParams),
  Time =  test_avg(labs_ops, evaluation, [Solution, SimParams], 50),
  BinSol = erlang:list_to_binary(Solution),
  OpenClTime = test_avg(labs_bin_ops, evaluation, [BinSol, SimParams], 50),
  ?assertEqual(Time,
               OpenClTime).



%%** exception error: {assertEqual_failed,[{module,cl_integration_test},
%%                                         {line,58},
%%                                         {expression,"OpenClTime"},
%%                                         {expected,36635},
%%                                         {value,660785}]}

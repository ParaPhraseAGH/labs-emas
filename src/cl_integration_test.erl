-module(cl_integration_test).


-include_lib("eunit/include/eunit.hrl").

-include_lib("emas/include/emas.hrl").

run_test() ->
  SimParams = #sim_params{ problem_size = 1024},
  Solution = labs_ops:solution(SimParams),
  Energy = labs_ops:energy(Solution),
  io:format(user, ">>> Energy: ~p~n", [Energy]).




same_energy_test() ->
  SimParams = #sim_params{ problem_size = 60},
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
  SimParams = #sim_params{ problem_size = 1024  },
  Solution = labs_ops:solution(SimParams),
  Time =  test_avg(labs_ops, energy, [Solution], 50),
  BinSol = erlang:list_to_binary(Solution),
  OpenClTime = test_avg(labs_bin_ops, energy, [BinSol], 50),
  ?assertEqual(Time,
               OpenClTime).



%%** exception error: {assertEqual_failed,[{module,cl_integration_test},
%%                                         {line,58},
%%                                         {expression,"OpenClTime"},
%%                                         {expected,36635},
%%                                         {value,660785}]}




  

  

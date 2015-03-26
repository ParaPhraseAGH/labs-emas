-module (labs_bin_ops).
-behaviour (emas_genetic_ops).

-ifdef(EUNIT).
-compile(export_all).
-endif.

-include_lib("emas/include/emas.hrl").

-export ([solution/1,
          evaluation/2,
          mutation/2,
          recombination/3,
          config/1,
          cleanup/1]).

-type sim_params() :: emas:sim_params().
-type solution() :: emas:solution(binary()).


%% @doc Generates a random solution.
-spec solution(sim_params()) -> solution().
solution(SP) ->
  << << ( random:uniform(2) - 1 ) >>
     || _ <- lists:seq(1, SP#sim_params.problem_size) >>.


%% @doc Evaluates a given solution. Higher is better.
-spec evaluation(solution(), sim_params()) -> float().
evaluation(Solution, SP) ->
  energy(Solution, SP).

-spec energy(solution(), sim_params()) -> float().
energy(S, SP) ->
  Pid = SP#sim_params.extra,
  cl_ops:energy(Pid, S).

-spec recombination(solution(), solution(), sim_params()) ->
                       {solution(), solution()}.
recombination(S1, S2, _SP) ->
  Zipped = [recombination_features(F1, F2) || {F1, F2} <- binzip(S1, S2)],
  binunzip(Zipped).

%% @doc Chooses a random order between the two initial features.
-spec recombination_features(any(), any()) -> {any(), any()}.
recombination_features(F, F) -> {F, F};
recombination_features(F1, F2) ->
  case random:uniform() < 0.5 of
    true -> {F1, F2};
    false -> {F2, F1}
  end.

-compile(export_all).

-spec binzip(binary(), binary()) -> [{0 | 1, 0 | 1}].
binzip(X, Y) ->
  binzip(X, Y, []).

binzip(<<>>, <<>>, Acc) ->
  Acc;
binzip(<<X:8/bits, XS/binary>>, <<Y:8/bits, YS/binary>>, Acc) ->
  binzip(XS, YS, [{X, Y} | Acc]).

binunzip(A) ->
  binunzip(A, <<>>, <<>>).
binunzip([], X, Y) ->
  {X, Y};
binunzip([{H1, H2} | T], X, Y) ->
  binunzip(T, << H1/binary, X/binary>>, << H2/binary, Y/binary >>).



%% @doc Reproduction function for a single agent (mutation only).
-spec mutation(solution(), sim_params()) -> solution().
mutation(Solution, SP) ->
  << << (mutate_bin(B, SP)) >> || << B >> <= Solution >>.

mutate_bin(X, SP) ->
  case random:uniform() < SP#sim_params.mutation_rate of
    true -> fnot(X);
    _ -> X
  end.

-spec config(sim_params()) -> term().
config(SP) ->
  Pid = cl_ops:start(),
  io:format("Pid ~p~n",[Pid]),
  Pid.

-spec cleanup(sim_params()) -> term().
cleanup(SP) ->
  Pid = SP#sim_params.extra,
  cl_ops:stop(Pid),
  ok.


%% internal functions

fnot(X) -> -X + 1.

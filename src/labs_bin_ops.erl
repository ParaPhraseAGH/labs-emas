-module (labs_bin_ops).
-behaviour (emas_genetic_ops).

-ifdef(EUNIT).
-compile(export_all).
-endif.

-include_lib("emas/include/emas.hrl").
-include_lib("cl/include/cl.hrl").

-export ([solution/1, evaluation/2, mutation/2, recombination/3, config/0, energy/1]).

-type sim_params() :: emas:sim_params().
-type solution() :: emas:solution(binary()).


%% @doc Generates a random solution.
-spec solution(sim_params()) -> solution().
solution(SP) ->
  << << ( random:uniform(2) - 1 ) >>
     || _ <- lists:seq(1, SP#sim_params.problem_size) >>.


%% @doc Evaluates a given solution. Higher is better.
-spec evaluation(solution(), sim_params()) -> float().
evaluation(Solution, _SP) ->
  energy(Solution).
%% local_search(Solution).

source() ->
  "
__kernel void square( __global char* input,
                      __global char* output,
                      const unsigned int count)
    {
      int i = get_global_id(0);
        if (i < count)
           output[i] = input[i]*input[i]*i;
           }
    ".

dump_data(Bin) ->
             io:format("data=~p\n", [[ X || <<X:32/native-float>> <= Bin ]]).

-spec energy(solution()) -> float().
energy(S) ->
  erlang:display_string("enqueu write\n"),

  
  E = clu:setup(all),
  io:format("platform created\n"),
  {ok,Program} = clu:build_source(E, source()),
  io:format("program built\n"),

  N = byte_size(S), %% number of bytes in indata
  Count = byte_size(S),     %% number of floats in indata


  %% Create input data memory (implicit copy_host_ptr)
  {ok,Input} = cl:create_buffer(E#cl.context,[read_only],N),
  io:format("input memory created\n"),

  %% Create the output memory
  {ok,Output} = cl:create_buffer(E#cl.context,[write_only],N),
  io:format("output memory created\n"),

  %% Create the command queue for the first device
  {ok,Queue} = cl:create_queue(E#cl.context,hd(E#cl.devices),[]),
  io:format("queue created\n"),

  %% Create the squre kernel object
  {ok,Kernel} = cl:create_kernel(Program, "square"),
  io:format("kernel created: ~p\n", [Kernel]),

  clu:apply_kernel_args(Kernel, [Input, Output, Count]),
  io:format("kernel args set\n"),

  %% Write data into input array
  {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, N, S, []),
  io:format("write data enqueued\n"),


  Device = hd(E#cl.devices),
  {ok,Local} = cl:get_kernel_workgroup_info(Kernel, Device, work_group_size),
  io:format("work_group_size = ~p\n", [Local]),

  %% Enqueue the kernel
  Global = Count,

  io:format(">>> Global = ~p\n", [Global]),
  {ok,Event2} = cl:enqueue_nd_range_kernel(Queue, Kernel,
                                           [Global], [Local], [Event1]),
  io:format("nd range [~p, ~p] kernel enqueued\n",
            [[Global],[Local]]),

  %% Enqueue the read from device memory (wait for kernel to finish)
  {ok,Event3} = cl:enqueue_read_buffer(Queue,Output,0,N,[Event2]),
  io:format("read buffer enqueued\n"),

  %% Now flush the queue to make things happend
  ok = cl:flush(Queue),
  io:format("flushed\n"),

  %% Wait for Result buffer to be written
  io:format("wait\n"),
  io:format("Event1 = ~p\n", [cl:wait(Event1)]),
  io:format("Event2 = ~p\n", [cl:wait(Event2)]),
  Event3Res = cl:wait(Event3),
  io:format(user,"Event3 = ~p\n", [Event3Res]),

  %%

  cl:release_mem_object(Input),
  cl:release_mem_object(Output),
  cl:release_queue(Queue),
  cl:release_kernel(Kernel),
  cl:release_program(Program),


  clu:teardown(E),
  {ok,EventResData} = Event3Res,
  dump_data(EventResData),

  List = erlang:binary_to_list(S),
  labs_ops:energy(List).







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

-spec config() -> term().
config() ->
  undefined.

%% internal functions

drop([], _) -> [];
drop(L, 0) -> L;
drop([_ | T], N) ->
  drop(T, N - 1).

foldzip(A, B) -> foldzip(A, B, 0).

foldzip([], _, Acc) -> Acc;
foldzip(_, [], Acc) -> Acc;
foldzip([HA|TA], [HB|TB], Acc) ->
  foldzip(TA, TB, Acc + dot(HA, HB)).

dot(X, X) -> 1;
dot(_, _) -> -1.

fnot(X) -> -X + 1.

-spec local_search(solution()) -> float().
local_search(Solution) ->
  MaxIterations = 15,
  {_Sol, Eval} = local_search(MaxIterations, Solution, energy(Solution)),
  Eval.

-spec local_search(integer(), solution(), float()) -> {solution(), float()}.
local_search(0, Solution, Evaluation) ->
  {Solution, Evaluation};
local_search(RemainingSteps, Solution, Evaluation) ->
  {BestSol, BestEval} = best_flipped(Solution),
  case BestEval > Evaluation of
    true -> local_search(RemainingSteps-1, BestSol, BestEval);
    _ -> {Solution, Evaluation}
  end.

best_flipped(Solution) ->
  FlippedSols = lists:map(fun (I) -> flip_nth(Solution, I) end,
                          lists:seq(1, length(Solution))),
  First = hd(FlippedSols),
  InitAcc = {First, energy(First)},
  GetBest = fun (S, {AccSol, AccE}) ->
                E = energy(S),
                case E > AccE of
                  true -> {S, E};
                  _ -> {AccSol, AccE}
                end
            end,
  lists:foldl(GetBest, InitAcc, FlippedSols).

flip_nth(Sol, N) ->
  flip_nth(Sol, [], N).

flip_nth([HS | TS], Acc, 1) ->
  lists:reverse(Acc) ++ [fnot(HS) | TS];
flip_nth([HS | TS], Acc, N) ->
  flip_nth(TS, [HS | Acc], N-1).

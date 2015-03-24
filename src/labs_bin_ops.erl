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

source() ->
  {ok, Binary} = file:read_file("src/energy.cl"),
  Binary.

reduce_source() ->
  {ok, Binary} = file:read_file("src/reduce.cl"),
  Binary.

-spec energy(solution()) -> float().
energy(S) ->
  E = clu:setup(all),
  %% io:format("platform created: ~p\n",[E]),

  Size = byte_size(S),  %% number of points in indata

  %% Calculate problem size
  Local = multiply_of_two_greater_than(Size),
  %% io:format("work_group_size = ~p\n", [Local]),

  Global = Local * (Size + 1),
  %% io:format("Global = ~p\n", [Global]),

  %% Create input data memory (implicit copy_host_ptr)
  {ok,Input} = cl:create_buffer(E#cl.context,[read_write],byte_size(S)),
  %% io:format("input memory created\n"),

  {ok,Fitness} = cl:create_buffer(E#cl.context,[read_write],(Size + 1) * 8),

  %% Create the output memory
  FloatBytesCount = 8,
  {ok,Output} = cl:create_buffer(E#cl.context,[write_only], FloatBytesCount),
  %% io:format("output memory created\n"),

  %% Create the command queue for the first device
  {ok,Queue} = cl:create_queue(E#cl.context,hd(E#cl.devices),[]),
  %% io:format("queue created\n"),

  %% Write data into input array
  {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, Size, S, []),
  %% io:format("write data enqueued\n"),

  {ok,FitnessProgram} = clu:build_source(E, source()),
  %% io:format("program built\n"),

  %% Create the squre kernel object
  {ok,FitnessKernel} = cl:create_kernel(FitnessProgram, "energy"),
  %% io:format("kernel created: ~p\n", [Kernel]),

  clu:apply_kernel_args(FitnessKernel, [Input,
                                        {local, Size}, % mutatedAgent
                                        {local, Local * 4 }, % int colerations
                                        Fitness,
                                        Size]), % size

  %% io:format("kernel args set\n"),

  {ok,Event2} = cl:enqueue_nd_range_kernel(Queue, FitnessKernel,
                                           [Global], [Local], [Event1]),
  %% io:format("nd range [~p, ~p] kernel enqueued\n", [[Global],[Local]]),

  {ok, ReduceProgram} = clu:build_source(E, reduce_source()),

  {ok, ReduceKernel} = cl:create_kernel(ReduceProgram, "reduce"),

  clu:apply_kernel_args(ReduceKernel, [Input,
                                       Output,
                                       Fitness,
                                       {local, Local * 8}, % doble localFitness
                                       {local, Local * 4}, % int indexes
                                       Size]), % size

  {ok,Event3} = cl:enqueue_nd_range_kernel(Queue, ReduceKernel,
                                           [Local], [Local], [Event2]),

  %% Enqueue the read from device memory (wait for kernel to finish)
  {ok,Event4} = cl:enqueue_read_buffer(Queue,Output,0,FloatBytesCount,[Event3]),
  %% io:format("read buffer enqueued\n"),

  %% Now flush the queue to make things happend
  ok = cl:flush(Queue),
  %% io:format("flushed\n"),

  %% Wait for Result buffer to be written
  %% io:format("wait\n"),
  %% io:format("Event1 = ~p\n", [cl:wait(Event1)]),
  %% io:format("Event2 = ~p\n", [cl:wait(Event2)]),
  Event3Res = cl:wait(Event4),
  %% io:format("Event3 = ~p\n", [Event3Res]),
  {ok, <<Energy:64/float-native>>} = Event3Res,
  %% io:format(">>> Energy = ~p\n", [Energy]),

  %% CleanUp
  cl:release_mem_object(Input),
  cl:release_mem_object(Fitness),
  cl:release_mem_object(Output),
  cl:release_queue(Queue),
  cl:release_kernel(FitnessKernel),
  cl:release_program(FitnessProgram),

  clu:teardown(E),

  Energy.


multiply_of_two_greater_than(Number) ->
  multiply_of_two_greater_than(Number, 2).

multiply_of_two_greater_than(Number, Multiply) when Multiply >= Number ->
  Multiply;
multiply_of_two_greater_than(Number, Multiply) ->
  multiply_of_two_greater_than(Number, Multiply * 2).


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

fnot(X) -> -X + 1.


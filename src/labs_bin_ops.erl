-module (labs_bin_ops).
-behaviour (emas_genetic_ops).

-ifdef(EUNIT).
-compile(export_all).
-endif.

-include_lib("emas/include/emas.hrl").
-include_lib("cl/include/cl.hrl").

-export ([solution/1, evaluation/2, mutation/2, recombination/3, config/0]).

-type sim_params() :: emas:sim_params().
-type solution() :: emas:solution(binary()).

-record (ocl, { context,
                devices,
                fit_kernel,
                red_kernel
              }).



%% @doc Generates a random solution.
-spec solution(sim_params()) -> solution().
solution(SP) ->
  << << ( random:uniform(2) - 1 ) >>
     || _ <- lists:seq(1, SP#sim_params.problem_size) >>.


%% @doc Evaluates a given solution. Higher is better.
-spec evaluation(solution(), sim_params()) -> float().
evaluation(Solution, SP) ->
  energy(Solution, SP).

source() ->
  {ok, Binary} = file:read_file("src/energy.cl"),
  Binary.

reduce_source() ->
  {ok, Binary} = file:read_file("src/reduce.cl"),
  Binary.

-spec energy(solution(), sim_params()) -> float().
energy(S, SP) ->
  #ocl{ context = Context,
        devices = Devices,
        fit_kernel = FitnessKernel,
        red_kernel = ReduceKernel
      } = SP#sim_params.extra,


  {ok,Queue} = cl:create_queue(Context,hd(Devices),[]),

  %% Create the fitness kernel object

  %% Calculate problem size
  Size = byte_size(S),  %% number of points in indata
  Local = multiply_of_two_greater_than(Size),
  Global = Local * (Size + 1),

  %% Create input data memory (implicit copy_host_ptr)
  FloatSize = 4, % in bytes
  IntSize = 4, % in bytes

  {ok,Input} = cl:create_buffer(Context,[read_write],Size),
  {ok,Fitness} = cl:create_buffer(Context,[read_write],(Size + 1) * FloatSize),
  {ok,Output} = cl:create_buffer(Context,[write_only], FloatSize), % one float
  

  clu:apply_kernel_args(FitnessKernel, [Input,
                                        {local, Size}, % mutatedAgent
                                        {local, Local * IntSize}, % int colerations
                                        Fitness,
                                        Size]), % size

  clu:apply_kernel_args(ReduceKernel, [Input,
                                       Output,
                                       Fitness,
                                       {local, Local * FloatSize}, % doble localFitness
                                       {local, Local * IntSize}, % int indexes
                                       Size]), % size

  %% Write data into input array
  {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, Size, S, []),

  %% enqueue kernels
  Event2 = enqueue_kernels(Queue, FitnessKernel, ReduceKernel, Global, Local, Event1, 15),

  %% Wait for Result buffer to be written
  FloatSize = 4,
  {ok,Event3} = cl:enqueue_read_buffer(Queue,Output,0,FloatSize,[Event2]),

  %% Now flush the queue to make things happend
  ok = cl:flush(Queue),

  Event3Res = cl:wait(Event3),
  {ok, <<Energy:32/float-native>>} = Event3Res,

  cl:release_mem_object(Input),
  cl:release_mem_object(Fitness),
  cl:release_mem_object(Output),
  
  cl:release_queue(Queue),

  Energy.


multiply_of_two_greater_than(Number) ->
  multiply_of_two_greater_than(Number, 2).

multiply_of_two_greater_than(Number, Multiply) when Multiply >= Number ->
  Multiply;
multiply_of_two_greater_than(Number, Multiply) ->
  multiply_of_two_greater_than(Number, Multiply * 2).


enqueue_kernels(_Queue, _FitnessKernel, _ReduceKernel, _Global, _Local, Event, 0) ->
  Event;
enqueue_kernels(Queue, FitnessKernel, ReduceKernel, Global, Local, Event, Count) when
    Count > 0  ->
  {ok,Event1} = cl:enqueue_nd_range_kernel(Queue, FitnessKernel,
                                           [Global], [Local], [Event]),
  ok = cl:enqueue_barrier(Queue),
  {ok,Event2} = cl:enqueue_nd_range_kernel(Queue, ReduceKernel,
                                           [Local], [Local], [Event1]),
  ok = cl:enqueue_barrier(Queue),

  enqueue_kernels(Queue, FitnessKernel, ReduceKernel, Global, Local, Event2, Count - 1).



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
  E = clu:setup(all),

  %% Create the fitness kernel object
  {ok,FitnessProgram} = clu:build_source(E, source()),
  {ok,FitnessKernel} = cl:create_kernel(FitnessProgram, "energy"),

  {ok, ReduceProgram} = clu:build_source(E, reduce_source()),
  {ok, ReduceKernel} = cl:create_kernel(ReduceProgram, "reduce"),


  %% CleanUp
  %% cl:release_kernel(FitnessKernel),
  %% cl:release_program(FitnessProgram),
  %% clu:teardown(E),

  #ocl{ context = E#cl.context,
        devices = E#cl.devices,
        fit_kernel = FitnessKernel,
        red_kernel = ReduceKernel
      }.






%% internal functions

fnot(X) -> -X + 1.

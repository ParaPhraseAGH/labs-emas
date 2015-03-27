-module(cl_ops).

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%
%% Calculate Fitness using OpenCL
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% API
-export([start/1,
         energy/2,
         stop/1]).

%% internal
-export([init/1,
         print_message_queue/1]).

-include_lib("cl/include/cl.hrl").

-record (ocl, { enviroment,
                context,
                devices,
                queue,
                fit_program,
                fit_kernel,
                red_program,
                red_kernel,
                input,
                fitness,
                output,
                size,
                local,
                global_size,
                timer
              }).


%% API %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

start(Size) ->
  proc_lib:spawn_link(?MODULE, init,[Size]).


energy(Pid, Solution) ->
  Pid ! {calculate, Solution, self()},
  receive {energy, Energy} ->
      Energy
  after 1000 ->
      erlang:error(timeout)
  end.


stop(Pid)->
  Pid ! stop.


%% internal %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% keep track of message queue length
print_message_queue(Pid) ->
      io:format(">>> pid(): ~p; queue: ~p~n",
                [Pid,
                 erlang:process_info(Pid, message_queue_len)]).

init(Size) ->

  {ok,Timer} = timer:apply_interval(500, ?MODULE, print_message_queue, [self()]),

  E = clu:setup(all),
  Context = E#cl.context,
  Devices = E#cl.devices,

  %% Create the fitness kernel object
  {ok,FitnessProgram} = clu:build_source(E, source()),
  {ok,FitnessKernel} = cl:create_kernel(FitnessProgram, "energy"),

  {ok, ReduceProgram} = clu:build_source(E, reduce_source()),
  {ok, ReduceKernel} = cl:create_kernel(ReduceProgram, "reduce"),

  {ok,Queue} = cl:create_queue(Context,hd(Devices),[]),

  %% Create the fitness kernel object

  %% Calculate problem size
  Local = multiply_of_two_greater_than(Size),
  Global = Local * (Size + 1),

  %% Create input data memory (implicit copy_host_ptr)
  FloatSize = 4, % in bytes

  {ok,Input} = cl:create_buffer(Context,[read_write],Size),
  {ok,Fitness} = cl:create_buffer(Context,[read_write],(Size + 1) * FloatSize),
  {ok,Output} = cl:create_buffer(Context,[write_only], FloatSize), % one float

  OCL =#ocl{ enviroment = E,
             context = Context,
             devices = Devices,
             queue = Queue,
             fit_program = FitnessProgram,
             fit_kernel = FitnessKernel,
             red_program = ReduceProgram,
             red_kernel = ReduceKernel,
             input = Input,
             fitness = Fitness,
             output = Output,
             size = Size,
             local = Local,
             global_size = Global,
             timer = Timer
            },
  loop(OCL).

source() ->
  {ok, Binary} = file:read_file("src/energy.cl"),
  Binary.


reduce_source() ->
  {ok, Binary} = file:read_file("src/reduce.cl"),
  Binary.



loop(State) ->
  receive
    {calculate, Solution, From} ->
      NewState = handle_calculate(Solution, From, State),
      loop(NewState);
    stop ->
      flush_loop(State)
  end.

%% calcuates all messages from inbox, and than stops
flush_loop(State) ->
  receive
    {calculate, Solution, From} ->
      NewState = handle_calculate(Solution, From, State),
      flush_loop(NewState)
  after 0 ->
      terminate(State)
  end.



handle_calculate(Solution, From, State) ->
  #ocl{ fit_kernel = FitnessKernel,
        red_kernel = ReduceKernel,
        queue = Queue,
        input = Input,
        fitness = Fitness,
        output = Output,
        size = Size,
        local = Local,
        global_size = Global
      } = State,

  IntSize = 4, % in bytes
  FloatSize = 4, % in bytes

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
  {ok,Event1} = cl:enqueue_write_buffer(Queue, Input, 0, Size, Solution, []),

  %% enqueue kernels
  Event2 = enqueue_kernels(Queue, FitnessKernel, ReduceKernel, Global, Local, Event1, 15),

  %% Wait for Result buffer to be written
  FloatSize = 4,
  {ok,Event3} = cl:enqueue_read_buffer(Queue,Output,0,FloatSize,[Event2]),

  %% Now flush the queue to make things happend
  ok = cl:flush(Queue),

  Event3Res = cl:wait(Event3),
  {ok, <<Energy:32/float-native>>} = Event3Res,

  From ! {energy, Energy},
  State.

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


terminate(#ocl{ enviroment = E,
                fit_program = FitnessProgram,
                fit_kernel = FitnessKernel,
                red_program = ReduceProgram,
                red_kernel = ReduceKernel,
                queue = Queue,
                input = Input,
                fitness = Fitness,
                output = Output,
                timer = Timer}) ->
  timer:cancel(Timer),
  %% CleanUp
  cl:release_mem_object(Input),
  cl:release_mem_object(Fitness),
  cl:release_mem_object(Output),

  cl:release_queue(Queue),

  cl:release_kernel(FitnessKernel),
  cl:release_program(FitnessProgram),
  cl:release_kernel(ReduceKernel),
  cl:release_program(ReduceProgram),
  clu:teardown(E),
  ok.

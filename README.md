# Concurrency vs Parallelism:

Concurrency is about dealing with multiple tasks at once and is achieved through techniques that make it seem like tasks 
are running simultaneously when in fact they are not. <br/>

Parallelism is about doing multiple tasks simultaneously, requiring multiple processing units for true simultaneous execution. <br/>

Cooperative and preemptive multitasking are two distinct methods used by operating systems to manage concurrency:
1. Cooperative: Each process controls the CPU for as long as it needs without being preempted by the operating system. Old Windows used to do it.
2. Preemptive multitasking allows the operating system to control the CPU scheduling decisively. Linux always used this.

Throughput is defined as the rate of doing work or how much work gets done per unit of time. If you are an Instagram user, 
you could define throughput as the number of images downloaded by your phone or browser per unit of time. <br/>

Latency is defined as the time required to complete a task or produce a result. Latency is also referred to as response time. 
The time it takes for a web browser to download Instagram images from the internet is the latency for downloading the images. <br/>

Synchronous execution refers to line-by-line execution of code. If a function is invoked, the program execution waits until the function call is completed. <br/>
An asynchronous program doesn’t wait for a task to complete before moving on to the next task. <br/>

Whenever threads are introduced in a program, the shared state amongst the threads becomes vulnerable to corruption.
```python
count = 0

def increment():
    global count
    count += 1
```
In this code, the count += 1 is not an atomic operation, which means that it has multiple steps/code before the count value
is incremented which makes it vulnerable to corruption when two or more threads try to update it.
This is the reason we apply locks to the sections of the code which need to be sequentially executed.

Critical section is any piece of code that has the possibility of being executed concurrently by more than one thread 
of the application and exposes any shared data or resources used by the application for access. <br/>

Deadlocks occur when two or more threads aren't able to make any progress because the resource required by the first thread 
is held by the second and the resource required by the second thread is held by the first. <br/>

A live-lock occurs when two threads continuously react in response to the actions by the other thread without making any real progress. <br/>

Other than a deadlock, an application thread can also experience starvation when it never gets CPU time or access to shared resources <br/>
An RLock or reentrant lock allows the same thread to acquire the lock multiple times without causing a deadlock (useful in case of nested/recursive functions), and it must
also release the lock the same number of times it acquired.<br/>

#### Mutex:
Mutex as the name hints implies mutual exclusion. A mutex is used to guard shared data such as a linked-list, an array, or any primitive type. 
A mutex allows only a single thread to access a resource or critical section.
Once a thread acquires a mutex, all other threads attempting to acquire the same mutex are blocked until the first thread releases the mutex. <br/>

#### Semaphores:
Semaphore, on the other hand, is used for limiting access to a collection of resources. Think of semaphore as having a limited number of permits 
to give out. If a semaphore has given out all the permits it has, then any new thread that comes along requesting a permit will be blocked till 
an earlier thread with a permit returns it to the semaphore. <br/>

#### Monitor:
A monitor is a synchronization construct that helps manage access to shared resources by multiple threads in a concurrent programming environment. <br/>
Key components of a monitor include:
1. Mutex (Lock): A lock that ensures mutual exclusion, i.e., at most one thread can execute within the monitor at any given time.
2. Condition Variables: These allow threads to wait for certain conditions to be true within the monitor. A thread waiting on a condition variable is suspended until another thread signals the condition variable, indicating that the condition has been met. <br/>

Condition variables are the way for threads to communicate with each other, it works like a producer-consumer problem.
Example: Finding prime number, we have a finder function and a printer function.

PRINTER THREAD WAITS UNTIL THERE IS SOMETHING TO PRINT.
cond_var.acquire()
while not found_prime and not exit_prog:
    cond_var.wait()
...print prime number
cond_var.release()

Let's say thread1 goes to wait state. Now another thread2 executes and then finds the prime number and calls the notify method of the same condition variable.
This 'notify' method will wake up the thread1.

FINDER THREAD FINDS AND NOTIFIES THE PRINTER THREAD
cond_var.acquire()
found_prime = True
cond_var.notify()
cond_var.release()

#### notify_all():
notify_all() method can be used when there is more than one thread waiting on a condition variable. It can also be used if there's a single thread waiting. The sequence of events on a notify_all() when multiple threads are waiting is described below:

1. A thread comes along acquires the lock associated with the condition variable, and calls wait()
2. The thread invoking wait() gives up the lock and goes to sleep or is taken off the CPU timeslice
3. The given up lock can be reacquired by a second thread that then too calls wait(), gives up the lock, and goes to sleep.
4. Notice that the lock is available for any other thread to acquire and either invoke a wait or a notify on the associated condition variable.
5. Another thread comes along acquires the lock and invokes notify_all() and subsequently releases the lock.
6. Note it is imperative to release the lock, otherwise the waiting threads can't reacquire the lock and return from the wait() call.
7. The waiting threads are all woken up but only one of them gets to acquire the lock. This thread returns from the wait() method and proceeds forward. The thread selected to acquire the lock is random and not in the order in which threads invoked wait().
8. Once the thread that is the first to wake up and make progress releases the lock, other threads acquire the lock one by one and proceed ahead.

#### Global Intepreter Lock:
The Python Global Interpreter Lock (GIL) is a mutex that allows only one thread to execute in the Python interpreter at any given time. 
This lock is necessary because Python's memory management is not thread-safe by default.
To mitigate the limitations of the GIL, Python developers often use multiprocessing instead of multithreading for concurrent 
execution, as each Python process has its own separate GIL and interpreter, allowing them to run truly in parallel on multiple cores. <br/>

Daemon threads run in the background, and it may continue to run even if the main program exits.
Main threads cannot continue to run once the program exits.

#### Barrier:
A barrier is a synchronization construct to wait for a certain number of threads to reach a common synchronization point in code. 
The involved threads each invoke the barrier object's wait() method and get blocked till all of threads have called wait(). When 
the last thread invokes wait() all of the waiting threads are released simultaneously. <br/>
```python
from threading import Barrier
from threading import Thread
from threading import current_thread
import random
import time


def thread_task():
    time.sleep(random.randint(0, 5))
    print("\nCurrently {0} threads blocked on barrier".format(barrier.n_waiting))
    barrier.wait()


def when_all_threads_released():
    print("All threads released, reported by {0}".format(current_thread().getName()))


num_threads = 5
barrier = Barrier(num_threads, action=when_all_threads_released)
threads = [0] * num_threads

for i in range(num_threads):
    threads[i - 1] = Thread(target=thread_task)

for i in range(num_threads):
    threads[i].start()
```

#### Timer thread:
Executes a callback after a given duration in a thread:
```python
from threading import Timer

def timer_task():
    print("timer task")

timer = Timer(5, timer_task)
timer.start()
```

### Multiprocess:
The multiprocessing module offers the method set_start_method() to let the developer choose the way new processes are created. 
There are three of them:
1. fork
2. spawn
3. fork-server

Ex:
```python
import multiprocessing

if __name__ == '__main__':
    multiprocessing.set_start_method('fork')

    # change the value of Test.value before creating
    # a new process
    process = multiprocessing.Process(target=process_task, name="process-1")
    process.start()
    process.join()
```

Fork: fork() creates an exact copy of the parent's address space, including the program counter, variables, and so on. 
This means the child process starts executing the same program as the parent but can be used to run a different branch of code based on the return value of fork(). 
Not everything is copied to the child process when a fork happens. <br/>

Spawn: The spawn() function is used to launch a new process with a specified command. Unlike fork(), spawn() does not necessarily 
create a process that runs the same program as the parent. Instead, it can run any command or executable in the system's environment.
The parent process can communicate with the spawned child process through stdin, stdout, and stderr streams.<br/>

There are two ways that processes can communicate between themselves:
1. Queues
2. Pipes

Queues:
Queues in Python's multiprocessing context are thread and process-safe, meaning they can be used by multiple producers and 
consumers across different threads and processes without risking data corruption. They are implemented as FIFO (first-in, first-out) 
data structures, making them suitable for tasks where order needs to be preserved. <br/>
```python
from multiprocessing import Process, Queue, current_process
import multiprocessing
import random


def child_process(q):
    count = 0
    while not q.empty():
        print(q.get())
        count += 1

    print("child process {0} processed {1} items from the queue".format(current_process().name, count), flush=True)

if __name__ == '__main__':
    multiprocessing.set_start_method("forkserver")
    q = Queue()
    print("This machine has {0} CPUs".format(str(multiprocessing.cpu_count())))
    
    random.seed()
    for _ in range(100):
        q.put(random.randrange(10))

    p1 = Process(target=child_process, args=(q,))
    p2 = Process(target=child_process, args=(q,))

    p1.start()
    p2.start()

    p1.join()
    p2.join()
```

Pipes: Pipes provide a simpler form of communication by allowing a unidirectional or bidirectional flow of information between two endpoints. 
Each end of the pipe can either send or receive data, depending on how the pipe is configured (duplex or simplex). Pipes are generally faster 
than queues because they involve less overhead and are best suited for simpler scenarios where only two processes need to communicate. <br/>
Example: recv_conn, send_conn = Pipe(duplex=False)
```python
from multiprocessing import Process, Pipe
import time

def child_process(conn):
    for i in range(0, 10):
        conn.send("hello " + str(i + 1))
    conn.close()


if __name__ == '__main__':
    parent_conn, child_conn = Pipe()
    p = Process(target=child_process, args=(child_conn,))
    p.start()
    time.sleep(3)

    for _ in range(0, 10):
        msg = parent_conn.recv()
        print(msg)

    parent_conn.close()
    p.join()
```

We can also share a variable, array, etc with the child process.
Ex: var = Value('I', 1), we can pass the 'var' as argument to the child process.

The Pool object consists of a group of processes that can receive tasks for execution.
```python
from multiprocessing import Pool
import os

def init(main_id):
    print("pool process with id {0} received a task from main process with id {1}".format(os.getpid(), main_id))


def square(x):
    return x * x

if __name__ == '__main__':
    main_process_id = os.getpid()

    pool = Pool(processes=1,
                initializer=init,
                initargs=(main_process_id,),
                maxtasksperchild=1)

    result = pool.apply(square, (3,))
    print(result)
```

#### Manager:
Python provides a way to share data between processes that may be running on different machines. The previous examples we 
saw of inter-process communication were restricted to a single machine. Using the Manager class we can share objects between 
processes running on the same machine or different machines. 
You can also provide an auth_key for securing the connection using the manager class.<br/>

```python
from multiprocessing.managers import BaseManager
from multiprocessing import Process
from threading import Thread
import time, random

def ProcessA(port_num):
    my_string = "hello World"
    manager = BaseManager(address=('127.0.0.1', port_num))
    manager.register('get_my_string', callable=lambda: my_string)
    server = manager.get_server()

    Thread(target=shutdown,args=(server,)).start()

    server.serve_forever()

def ProcessB(port_num):
    manager = BaseManager(address=('127.0.0.1', port_num))
    manager.register('get_my_string')
    manager.connect()
    proxy_my_string = manager.get_my_string()

    print("In ProcessB repr(proxy_my_string) = {0}".format(repr(proxy_my_string)))
    print("In ProcessB str(proxy_my_string) = {0}".format(str(proxy_my_string)))

    print(proxy_my_string)
    print(proxy_my_string.capitalize())
    print(proxy_my_string._callmethod("capitalize"))

def shutdown(server):
    time.sleep(3)
    server.stop_event.set()


if __name__ == '__main__':
    port_num = random.randint(10000, 60000)

    # Start another process which will access the shared string
    p1 = Process(target=ProcessA, args=(port_num,), name="ProcessA")
    p1.start()

    time.sleep(1)

    p2 = Process(target=ProcessB, args=(port_num,), name="ProcessB")
    p2.start()

    p1.join()
    p2.join()
```
#### Namespace:
Namespace is a type that can be registered with a SyncManager for sharing between processes. It doesn't have public methods but we can add writeable attributes to it. 
Think of namespace as a bulletin board, where attributes can be assigned by one process, and read by others.<br/>
```python
from multiprocessing.managers import SyncManager
from multiprocessing import Process
import multiprocessing

def process1(ns):
    print(ns.item)
    ns.item = "educative"


def process2(ns):
    print(ns.item)
    ns.item = "educative is awesome !"

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    # create a namespace
    manager = SyncManager(address=('', 55555))
    manager.start()
    shared_vars = manager.Namespace()  # manager.Namespace()
    shared_vars.item = "empty"
    # manager.register("get_namespace", callable=lambda: None)

    # create the first process
    p1 = Process(target=process1, args=(shared_vars,))
    p1.start()
    p1.join()

    # create the second process
    p2 = Process(target=process2, args=(shared_vars,))
    p2.start()
    p2.join()

    print(shared_vars.item)
```


### Concurrent Package:
#### PoolExecutors:
1. Threadpool executor
2. ProcessPool executor

```python
from concurrent.futures import ThreadPoolExecutor
from threading import current_thread

def say_hi(item):
    print("\nhi " + str(item) + " executed in thread id " + current_thread().name, flush=True)

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=10)
    lst = list()
    # We can also use map function
    for i in range(1, 10):
        lst.append(executor.submit(say_hi, "guest" + str(i)))

    for future in lst:
        future.result()

    executor.shutdown()
```

ProcessPool Executor using map function:
```python
from concurrent.futures import ProcessPoolExecutor
import os

def square(item):
    print("Executed in process with id " + str(os.getpid()), flush=True)
    return item * item

if __name__ == '__main__':
    executor = ProcessPoolExecutor(max_workers=10)
    # chunksize=1 means that each process will get 1 datapoint as arg.
    # For chunksize=5, we will send 5 args into a single process, which means only 2 process is required for the below
    # 10 arguments.
    it = executor.map(square, (1, 2, 3, 4, 5, 6, 7, 8, 9, 10), chunksize=1)
    # looping over futures
    for sq in it:
        print(sq)

    executor.shutdown()
```

You can think of Future as an entity that represents a deferred computation that may or may not have been completed. 
It is an object that represents the outcome of a computation to be completed in future. <br/>
Methods which are provided in future object:
```python
from concurrent.futures import ThreadPoolExecutor
import time

def my_special_callback(ftr):
    res = ftr.result()
    print("my_special_callback invoked " + str(res))

def square(item):
    # simulate a computation by sleeping
    time.sleep(5)
    return item * item

if __name__ == '__main__':
    executor = ThreadPoolExecutor(max_workers=10)

    future = executor.submit(square, 7)

    print("is running : " + str(future.running()))
    print("is done : " + str(future.done()))
    print("Attempt to cancel : " + str(future.cancel()))
    print("is cancelled : " + str(future.cancelled()))
    # To see any exception
    ex = future.exception()
    # We can also add a callback to be executed after the future is ready.
    future.add_done_callback(my_special_callback)

    executor.shutdown()
```
The concurrent.futures module provides two methods to wait for futures collectively. These are:
1. wait
2. as_completed

```python
from concurrent.futures import wait
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ProcessPoolExecutor

def square(item):
    return item * item

if __name__ == '__main__':
    lst = list()
    threadExecutor = ThreadPoolExecutor(max_workers=10)
    processExecutor = ProcessPoolExecutor(max_workers=10)

    for i in range(1, 6):
        lst.append(threadExecutor.submit(square, i))

    for i in range(6, 11):
        lst.append(processExecutor.submit(square, i))

    result = wait(lst, timeout=None, return_when='ALL_COMPLETED')

    print("completed futures count: " + str(len(result.done)) + " and uncompleted futures count: " +
          str(len(result.not_done)) + "\n")

    for ftr in result.done:
        print(ftr.result())

    threadExecutor.shutdown()
    processExecutor.shutdown()
```
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
When you acquire a semaphore lock, the behavior depends on the state of the semaphore's internal counter. If the internal counter of the semaphore is 
greater than zero, the semaphore decrements the counter by one and allows the acquiring operation to proceed immediately without putting the thread or process to sleep. <br/>
However, if the internal counter is zero at the time of the acquisition attempt, this means that the maximum number of allowed concurrent operations 
(as defined by the semaphore's capacity) are already in progress. In this case, the semaphore will block the acquiring thread or process, effectively 
putting it to sleep until another thread or process releases the semaphore, incrementing the internal counter and allowing the blocked thread to proceed <br/>
```python
import threading
import time

# Function that represents the task each thread will execute
def task(semaphore, thread_number):
    with semaphore:
        print(f"Thread {thread_number} is running")
        time.sleep(2)  # Simulate a task taking some time to complete

# Create a semaphore that allows up to 3 threads to enter the critical section
semaphore = threading.Semaphore(3)
# List to hold the threads
threads = []
# Create and start 5 threads
for i in range(5):
    t = threading.Thread(target=task, args=(semaphore, i+1))
    threads.append(t)
    t.start()
# Wait for all threads to complete
for t in threads:
    t.join()
print("All threads have completed their tasks.")
```

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

### AsyncIO

A coroutine can be defined as a special function that can give up control to its caller without losing its state. The methods or functions 
that we are used to, the ones that conclusively return a value and don't remember state between invocations, can be thought of as a 
specialization of a coroutine, also known as subroutines. <br/>

The event loop is a programming construct that waits for events to happen and then dispatches them to an event handler.
Threads don't come cheap. Creating, maintaining and tearing down threads takes CPU cycles in addition to memory. In fact, this 
difference becomes more visible in webservers which use threads to handle HTTP web requests vs which use an event loop. 
Apache is an example of the former and NGINX of the latter. NGINX outshines Apache in memory usage under high load. <br/>

Native Coroutine can be defined as:
```python
import asyncio
async def coro():
    await asyncio.sleep(1)
```

asyncio provides a framework for dealing with asynchronous I/O tasks. It allows you to run multiple tasks and handle I/O operations without blocking the execution of your program. 
This is achieved through the use of coroutines, which are a type of function that can pause its execution before completing and can be resumed at a later point.
#### Key Components
1. Coroutines: Defined with async def. These are the functions you will execute asynchronously. They are used with await, which allows other tasks to run while waiting for an operation to complete.
2. Event Loop: Manages and distributes the execution of different tasks. It keeps track of all the running tasks and resumes their execution when the awaited operation is completed.
3. Tasks: These are used to schedule coroutines concurrently. When a coroutine is wrapped into a Task with functions like asyncio.create_task(), it’s scheduled to run on the event loop. <br/>

Example: Fetching data from urls:
```python
import asyncio

# Coroutine
async def fetch_data(url, delay):
    print(f"Starting to fetch data from {url}")
    await asyncio.sleep(delay)  # Simulate network delay
    print(f"Finished fetching data from {url}")
    return f"Data from {url}"

async def main():
    urls = ['url1', 'url2', 'url3']  # Simulated URLs
    delays = [2, 3, 1]  # Simulated network delays for each URL
    
    # Create tasks for each URL
    tasks = [fetch_data(url, delay) for url, delay in zip(urls, delays)]
    
    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks)
    
    # Optionally, process the results
    for result in results:
        print(result)

if __name__ == "__main__":
    # Run the event loop
    asyncio.run(main())

```

While asyncio is excellent for I/O-bound tasks, it's not beneficial for CPU-bound tasks. For such tasks, using multi-threading or multi-processing might be more appropriate. <br/>

### Problems:
Implementation of Blocking Queue:
```python
# Code to add item to the queue safely
def enqueue(self, item):
    self.cond.acquire()
    while self.curr_size == self.max_size:
        self.cond.wait()

    self.q.append(item)
    self.curr_size += 1
    # Notify all, such that the consumer who is trying to dequeue is wokenup
    self.cond.notifyAll()
    self.cond.release()

    # Code to remove item from the queue safely
def dequeue(self):
    self.cond.acquire()
    while self.curr_size == 0:
        self.cond.wait()

    item = self.q.pop(0)
    self.curr_size -= 1
    
    self.cond.notifyAll()
    self.cond.release()

    return item
```

#### Rate-Limit using token bucket:
We generate a token every second, but total number of available tokens cannot be greater than the max_token.
Every thread gets one token, we apply lock so that context-switching does not happen.
This mechanism ensures that the rate of requests does not exceed the number defined by MAX_TOKENS per second, 
effectively throttling the access to the resource the tokens are protecting. <br/>
```python
from threading import Thread
from threading import current_thread
from threading import Lock
import time

class TokenBucketFilter:
    def __init__(self, MAX_TOKENS):
        self.MAX_TOKENS = MAX_TOKENS
        self.last_request_time = time.time()
        self.possible_tokens = 0
        self.lock = Lock()
    def get_token(self):

        with self.lock:
            self.possible_tokens += int((time.time() - self.last_request_time))

            if self.possible_tokens > self.MAX_TOKENS:
                self.possible_tokens = self.MAX_TOKENS

            if self.possible_tokens == 0:
                time.sleep(1)
            else:
                self.possible_tokens -= 1

            self.last_request_time = time.time()

            print("Granting {0} token at {1} ".format(current_thread().getName(), int(time.time())))

if __name__ == "__main__":
    token_bucket_filter = TokenBucketFilter(1)
    threads = list()
    for _ in range(0, 10):
        threads.append(Thread(target=token_bucket_filter.get_token))

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
```
Another approach using daemon thread, where we start a daemon thread to fill in available tokens.
This approach uses condition variable to lock/notify.
```python
from threading import Thread
from threading import Condition
from threading import current_thread
import time

class MultithreadedTokenBucketFilter():
    def __init__(self, maxTokens):
        self.MAX_TOKENS = int(maxTokens)
        self.possibleTokens  = int(0)
        self.ONE_SECOND = int(1)
        self.cond = Condition()
        dt = Thread(target = self.daemonThread)
        dt.setDaemon(True)
        dt.start()
    
    def daemonThread(self):
        while True:
            self.cond.acquire()
            if self.possibleTokens < self.MAX_TOKENS:
                self.possibleTokens = self.possibleTokens + 1
            self.cond.notify() 
            self.cond.release()
            
            time.sleep(self.ONE_SECOND)
    
    def getToken(self):
        self.cond.acquire()
        while self.possibleTokens == 0:
            self.cond.wait()
        self.possibleTokens = self.possibleTokens - 1
        self.cond.release()

        print("Granting " + current_thread().getName() + " token at " + str(time.time()))

if __name__ == "__main__":
    threads_list = [];
    bucket = MultithreadedTokenBucketFilter(1)
    
    for x in range(10):
        workerthread =  Thread(target=bucket.getToken)
        workerthread.name = "Thread_" + str(x+1)
        threads_list.append(workerthread)
    
    for t in threads_list:
        t.start()
        
    for t in threads_list:
        t.join()
```

#### Thread Safe Deferred Callback:
Design and implement a thread-safe class that allows registration of callback methods that are executed after a user specified time interval in seconds has elapsed.

```python
from threading import Condition
from threading import Thread
import heapq
import time
import math

class DeferredCallbackExecutor:
    def __init__(self):
        self.actions = list()  # Maintaining list of actions 
        self.cond = Condition()
        self.sleep = 0
    
    def add_action(self, action):
        # add exec_at time for the action
        action.exec_secs_after = time.time() + action.exec_secs_after

        self.cond.acquire()
        # We use min heap to store the actions based on the earliest time, so that the action with the earliest time
        # is processed first.
        heapq.heappush(self.actions, action)
        self.cond.notify()
        self.cond.release()

    def start(self):

        while True:
            self.cond.acquire()
            
            # If list of actions is empty, we wait
            while len(self.actions) is 0:
                self.cond.wait()
            # If some actions exist, then we compute its to wait before execution,
            # if the time <= 0, its ready for execution and break, else we sleep
            while len(self.actions) is not 0:
                # calculate sleep duration
                next_action = self.actions[0]
                sleep_for = next_action.exec_secs_after - math.floor(time.time())
                if sleep_for <= 0:
                    # time to execute action
                    break

                self.cond.wait(timeout=sleep_for)
            
            # Getting the action from min heap
            action_to_execute_now = heapq.heappop(self.actions)
            # Executing the action.
            action_to_execute_now.action(*(action_to_execute_now,))

            self.cond.release()

# Actio object
class DeferredAction(object):
    def __init__(self, exec_secs_after, name, action):
        self.exec_secs_after = exec_secs_after
        self.action = action
        self.name = name

    def __lt__(self, other):
        return self.exec_secs_after < other.exec_secs_after

def say_hi(action):
        print("hi, I am {0} executed at {1} and required at {2}".format(action.name, math.floor(time.time()),
                                                                    math.floor(action.execute_at)))

if __name__ == "__main__":
    action1 = DeferredAction(3, ("A",), say_hi)
    action2 = DeferredAction(2, ("B",), say_hi)
    action3 = DeferredAction(1, ("C",), say_hi)
    action4 = DeferredAction(7, ("D",), say_hi)

    executor = DeferredCallbackExecutor()
    t = Thread(target=executor.start, daemon=True)
    t.start()

    executor.add_action(action1)
    executor.add_action(action2)
    executor.add_action(action3)
    executor.add_action(action4)

    # wait for all actions to execute
    time.sleep(15)
```

#### Read-Write Lock:
Imagine you have an application where you have multiple readers and a single writer. You are asked to design a lock which lets multiple readers read at the same time, but only one writer write at a time.
```python
from threading import Condition
from threading import Thread
from threading import current_thread
import time
import random

class ReadersWriteLock:

    def __init__(self):
        self.cond_var = Condition()
        self.write_in_progress = False
        self.readers = 0

    def acquire_read_lock(self):
        self.cond_var.acquire()

        while self.write_in_progress is True:
            self.cond_var.wait()

        self.readers += 1

        self.cond_var.release()

    def release_read_lock(self):
        self.cond_var.acquire()

        self.readers -= 1
        if self.readers is 0:
            self.cond_var.notifyAll()

        self.cond_var.release()

    def acquire_write_lock(self):
        self.cond_var.acquire()

        while self.readers is not 0 or self.write_in_progress is True:
            self.cond_var.wait()
        self.write_in_progress = True

        self.cond_var.release()

    def release_write_lock(self):
        self.cond_var.acquire()

        self.write_in_progress = False
        self.cond_var.notifyAll()

        self.cond_var.release()

def writer_thread(lock):
    while 1:
        lock.acquire_write_lock()
        print("\n{0} writing at {1} and current readers = {2}".format(current_thread().getName(), time.time(),
                                                                      lock.readers), flush=True)
        write_for = random.randint(1, 5)
        time.sleep(write_for)
        print("\n{0} releasing at {1} and current readers = {2}".format(current_thread().getName(), time.time(),
                                                                        lock.readers),
              flush=True)
        lock.release_write_lock()
        time.sleep(1)

def reader_thread(lock):
    while 1:
        lock.acquire_read_lock()
        print("\n{0} reading at {1} and write in progress = {2}".format(current_thread().getName(), time.time(),
                                                                        lock.write_in_progress), flush=True)
        read_for = random.randint(1, 2)
        time.sleep(read_for)
        print("\n{0} releasing at {1} and write in progress = {2}".format(current_thread().getName(), time.time(),
                                                                          lock.write_in_progress), flush=True)
        lock.release_read_lock()
        time.sleep(1)

if __name__ == "__main__":
    lock = ReadersWriteLock()
    writer1 = Thread(target=writer_thread, args=(lock,), name="writer-1", daemon=True)
    writer2 = Thread(target=writer_thread, args=(lock,), name="writer-2", daemon=True)

    writer1.start()

    readers = list()
    for i in range(0, 3):
        readers.append(Thread(target=reader_thread, args=(lock,), name="reader-{0}".format(i + 1), daemon=True))

    for reader in readers:
        reader.start()
    writer2.start()
    time.sleep(15)
```

#### Unisex-Bathroom Problem:
A bathroom is being designed for the use of both males and females in an office but requires the following constraints to be maintained:
1. There cannot be men and women in the bathroom at the same time.
2. There should never be more than three employees in the bathroom simultaneously.


```python
from threading import Semaphore
from threading import Condition
import time

class UnisexBathroomProblem:
    def __init__(self):
        self.in_use_by = "none"
        self.emps_in_bathroom = 0
        self.max_emps_sem = Semaphore(3)
        self.cond = Condition()

    def use_bathroom(self, name):
        # simulate using a bathroom
        print("\n{0} is using the bathroom. {1} employees in bathroom".format(name, self.emps_in_bathroom))
        time.sleep(1)
        print("\n{0} is done using the bathroom".format(name))

    def male_use_bathroom(self, name):
        # The with statement takes care of .acquire and .release of the condition variable.
        with self.cond:
            while self.in_use_by == "female":
                self.cond.wait()
            self.max_emps_sem.acquire()
            self.emps_in_bathroom += 1
            self.in_use_by = "male"

        self.use_bathroom(name)
        self.max_emps_sem.release()

        with self.cond:
            self.emps_in_bathroom -= 1
            if self.emps_in_bathroom == 0:
                self.in_use_by = "none"

            self.cond.notifyAll()

    def female_use_bathroom(self, name):
        with self.cond:
            while self.in_use_by == "male":
                self.cond.wait()

            self.max_emps_sem.acquire()
            self.emps_in_bathroom += 1
            self.in_use_by = "female"

        self.use_bathroom(name)
        self.max_emps_sem.release()

        with self.cond:
            self.emps_in_bathroom -= 1

            if self.emps_in_bathroom == 0:
                self.in_use_by = "none"

            self.cond.notifyAll()
```

### Uber Rider problem:
Imagine at the end of a political conference, republicans and democrats are trying to leave the venue and ordering Uber rides at the same time. 
However, to make sure no fight breaks out in an Uber ride, the software developers at Uber come up with an algorithm whereby either an Uber 
ride can have all democrats or republicans or two Democrats and two Republicans. All other combinations can result in a fist-fight. <br/>

```python
from threading import Semaphore
from threading import current_thread
from threading import Lock
from threading import Barrier

class UberSeatingProblem():
    def __init__(self):
        self.democrats_count = 0
        self.democrats_waiting = Semaphore(0)
        self.republicans_count = 0
        self.republicans_waiting = Semaphore(0)
        self.lock = Lock()
        self.barrier = Barrier(4)
        self.ride_count = 0

    def drive(self):
        self.ride_count += 1
        print("Uber ride # {0} filled and on its way".format(self.ride_count), flush=True)

    def seated(self, party):
        print("\n{0} {1} seated".format(party, current_thread().getName()), flush=True)

    def seat_democrat(self):
        ride_leader = False
        self.lock.acquire()
        self.democrats_count += 1
        if self.democrats_count == 4:
            # release 3 democrats to ride along
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            ride_leader = True
            self.democrats_count -= 4
        elif self.democrats_count == 2 and self.republicans_count >= 2:
            # release 1 democrat and 2 republicans
            self.democrats_waiting.release()
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            ride_leader = True
            # remember to decrement the count of dems and repubs
            # selected for next ride
            self.democrats_count -= 2
            self.republicans_count -= 2
        else:
            # can't form a valid combination, keep waiting and release lock
            self.lock.release()
            # Acquiring a semaphore means this rider is waiting.
            self.democrats_waiting.acquire()

        self.seated("Democrat")
        # Wait for all the 4 threads till this point for start driving.
        self.barrier.wait()
        if ride_leader is True:
            self.drive()
            self.lock.release()

    def seat_republican(self):
        ride_leader = False
        self.lock.acquire()
        self.republicans_count += 1
        if self.republicans_count == 4:
            # release 3 republicans to ride along
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            self.republicans_waiting.release()
            ride_leader = True
            self.republicans_count -= 4

        elif self.republicans_count == 2 and self.democrats_count >= 2:
            # release 1 republican and 2 democrats
            self.republicans_waiting.release()
            self.democrats_waiting.release()
            self.democrats_waiting.release()
            ride_leader = True

            # remember to decrement the count of dems and repubs
            # selected for next ride
            self.republicans_count -= 2
            self.democrats_count -= 2
        else:
            # can't form a valid combination, keep waiting and release lock
            self.lock.release()
            self.republicans_waiting.acquire()

        self.seated("Republican")
        self.barrier.wait()

        if ride_leader is True:
            self.drive()
            self.lock.release()
```
### Async to Sync problem:
This is an actual interview question asked at Netflix.

Imagine we have an AsyncExecutor class that performs some useful task asynchronously via the method execute(). 
In addition, the method accepts a function object that acts as a callback and gets invoked after the asynchronous execution is done.
The definition for the involved classes is below. The asynchronous work is simulated using sleep. A passed-in call is invoked to let 
the invoker take any desired action after the asynchronous processing is complete. <br/>

```python
from threading import Thread
from threading import Condition
from threading import current_thread
import time

class AsyncExecutor:
    def work(self, callback):
        time.sleep(5)
        callback()

    def execute(self, callback):
        # Here, self.work if called from SyncExecutor's instance will call the work method of the SyncExecutor because
        # its overridden.
        Thread(target=self.work, args=(callback,)).start()

class SyncExecutor(AsyncExecutor):
    def __init__(self):
        self.cv = Condition()
        self.is_done = False

    def work(self, callback):
        super().work(callback)
        print("{0} thread notifying".format(current_thread().name))
        self.cv.acquire()
        self.cv.notify_all()
        self.is_done = True
        self.cv.release()

    def execute(self, callback):
        super().execute(callback)
        self.cv.acquire()
        while self.is_done is False:
            self.cv.wait()
        print("{0} thread woken-up".format(current_thread().name))
        self.cv.release()

def say_hi():
    print("Hi")

if __name__ == "__main__":
    exec = SyncExecutor()
    exec.execute(say_hi)

    print("main thread exiting")

```

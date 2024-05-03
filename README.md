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
An RLock or reentrant lock allows the same thread to acquire the lock multiple times without causing a deadlock. <br/>

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


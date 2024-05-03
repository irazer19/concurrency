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


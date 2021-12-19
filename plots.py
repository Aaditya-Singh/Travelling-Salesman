def plot_qrtd(all_traces, opt_soln):
    # all_traces is a list of (lists of lists)
    num_runs = len(all_traces)
    quality_errors = [0.4, 0.6, 0.8, 1.0, 1.5, 2.0]
    plt.figure()
    
    
    for q_error in quality_errors:
        num_successful_runs = 0
        successful_runtimes = list()
        target = opt_soln*(1+q_error/100)
        #print("Target = ", target)
        
        # iterate through traces to see if soln is better than target
        for trace in all_traces:
            trace = np.array(trace)
            times = trace[:, 0]
            distances = trace[:, 1]
            #print("Times = ", times)
            #print("Distances = ", distances)

            if distances[-1]<=target:
                num_successful_runs += 1
                idx = np.argmax(distances<=target)
                #print("Index found = ", idx)
                successful_runtimes.append(times[idx])

        successful_runtimes = sorted(successful_runtimes)
        probs = [j/num_runs for j in range(num_successful_runs)]
        #print("Runtimes = ", successful_runtimes)
        #print("Probabilities =", probs)
        plt.plot(successful_runtimes, probs)
        
    
    legend = ["q*="+str(j) for j in quality_errors]
    plt.legend(legend)
    plt.title("QRTD plot")
    plt.xlabel("Runtimes")
    plt.ylabel("Probability of success")
    plt.show()
    
def plot_sqd(all_traces, opt_soln):
    
    # all_traces is a list of (lists of lists)
    num_runs = len(all_traces)
    cutoff_times = [0.2, 0.5, 1.0, 2.0, 3.0]
    plt.figure()
    
    
    for c_time in cutoff_times:
        
        q_values = list()
        
        # iterate through traces to see if soln is better than target
        for trace in all_traces:
            trace = np.array(trace)
            times = trace[:, 0]
            distances = trace[:, 1]
            #print("Times = ", times)
            #print("Distances = ", distances)

            
            idx = np.argmin(times<=c_time)
            #print("Index found = ", idx)
            soln_value = distances[idx]
            q_value = 100*(soln_value-opt_soln)/opt_soln
            q_values.append(q_value)
            

        q_values = sorted(q_values)
        probs = [j/num_runs for j in range(num_runs)]
        #print("q_values = ", q_values)
        #print("Probabilities =", probs)
        plt.plot(q_values, probs)
        
    
    legend = ["t="+str(j) for j in cutoff_times]
    plt.legend(legend)
    plt.title("SQD plot")
    plt.xlabel("q*")
    plt.ylabel("Probability of success")
    plt.show()

def boxplot(all_traces, opt_soln):
    
    # all_traces is a list of (lists of lists)
    num_runs = len(all_traces)
    quality_errors = [0.1, 0.2, 0.3, 0.4, 0.5]
    plt.figure(figsize=(10,8))
    
    
    for i, q_error in enumerate(quality_errors):
        num_successful_runs = 0
        successful_runtimes = list()
        target = opt_soln*(1+q_error/100)
        print("Target = ", target)
        
        # iterate through traces to see if soln is better than target
        for trace in all_traces:
            trace = np.array(trace)
            times = trace[:, 0]
            distances = trace[:, 1]
            print("Times = ", times)
            print("Distances = ", distances)

            if distances[-1]<=target:
                num_successful_runs += 1
                idx = np.argmax(distances<=target)
                print("Index found = ", idx)
                successful_runtimes.append(times[idx])

        print(successful_runtimes)
        plt.boxplot(successful_runtimes, positions=[q_error*10], notch = True, widths = 0.35, patch_artist = True, boxprops = dict(facecolor=str("C"+str(i))))
        
    
    #legend = ["q*="+str(j) for j in quality_errors]
    #plt.legend(legend)
    plt.title("Boxplot")
    plt.ylabel("Solving times")
    plt.xlabel("Solution Quality")
    plt.show()
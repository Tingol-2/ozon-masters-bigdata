from pyspark     import SparkContext, SparkConf
from collections import defaultdict

import sys

conf = SparkConf()
sc = SparkContext(appName="Shortest path", conf=conf)


start_node = int(sys.argv[1])
end_node   = int(sys.argv[2])

input_file  = sys.argv[3]
output_file = sys.argv[4]

graph = sc.textFile(input_file)
graph_data = graph.map(lambda x : x.split("\t")).map(lambda x : (int(x[1]), int(x[0]))).cache()

n_iter = 800
queue_s, visited_s = set(), set()

queue_s.add(start_node)
parent_s = defaultdict(list)
for i in range(n_iter):
    queue_sc, visited_sc = sc.broadcast(queue_s), sc.broadcast(visited_s)
    if (end_node in visited_s):
        break
    output = graph_data.filter(lambda x : (x[0] in queue_sc.value) and (x[0] not in visited_sc.value)).groupByKey().mapValues(list).collect()
    
    visited_s.update(queue_s)
    queue_s = set()
    for val in output:
        for item in val[1]:
            parent_s[item].append(val[0])
        queue_s.update(val[1])
path_length = i

def bfs(current, target, parents, level, max_level):
    if level >= max_level:
        return []
    if current == target:
        return [[target]]
    ret_val = list()
    for parent in parents[current]:
        temp_val = []
        for prev in bfs(parent,target, parents, level+1, max_level):
            temp_val.append(current)
            temp_val.extend(prev)
        if (len(temp_val) != 0):
            ret_val.append(temp_val)
    return ret_val

out = bfs(end_node, start_node, parent_s, 0, path_length)

result = list()
for st in out:
    path = list(st)
    path.reverse()
    result.append(','.join(map(lambda x :str(x), path)))

final_result = "\n".join(result)
result_rdd = sc.parallelize([final_result])
result_rdd.saveAsTextFile(output_file)

sc.stop()

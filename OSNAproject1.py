import praw
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics


###DATA COLLECTION AND PROCESSING###

# Connect to Reddit using PRAW
reddit = praw.Reddit(client_id='AOtDTmgh99EXMLnBvjXGZg',
                     client_secret='xAXNgiV3W21PbWfegkrlNhk6Yysysw',
                     username='jankii27',
                     password='Project_Password',
                     user_agent='Project1')

# Define the subreddit to crawl
subreddit = reddit.subreddit('Amazon')
set_comment_limit = 10

# Initialize lists to store data
nodeData = []
authorData = []

# Define the recursive_node_adder function
def recursive_node_adder(g, comment, parent_author):
    nodeItem = []
    # Check if the comment author exists
    if comment.author is not None:
        # Add the author to the authorData list if they haven't been added yet
        if comment.author not in authorData:
            authorData.append(comment.author)
        # Add the parent author and comment author to nodeItem
        nodeItem.append(parent_author)
        nodeItem.append(comment.author)
        # Iterate through the comment's replies
        for reply in comment.replies.list():
            # Skip if the reply is not a comment
            if isinstance(reply, praw.models.MoreComments):
                continue
            # Call the function recursively for each reply
            recursive_node_adder(g, reply, comment.author)
        # Add nodeItem to the nodeData list
        nodeData.append(nodeItem)

# Create an undirected graph using NetworkX
g = nx.Graph()

# Get the top 25 posts from the subreddit
submissions = subreddit.top(limit=25)

# Iterate through each post
for post in submissions:
    print(post.author, "-", post.title)
    # Check if the post author exists
    if post.author is not None:
        # Add the author to the authorData list if they haven't been added yet
        if post.author not in authorData:
            authorData.append(post.author)
    # Set the number of comments to retrieve for the post
    post.comment_limit = set_comment_limit
    # Iterate through each comment of the post
    for comment in post.comments.list():
        # Skip if the comment is not a comment
        if isinstance(comment, praw.models.MoreComments):
            continue
        # Call the recursive_node_adder function for each comment
        recursive_node_adder(g, comment, post.author)


# DataFrame
nodedf = pd.DataFrame(nodeData, columns=['Source', 'Target']).dropna()
nodedf.to_csv("nodedata.csv")
print("\nData:\n",nodedf.head(20),"\n\n")

graph_api = nx.from_pandas_edgelist(nodedf, source="Source", target="Target",
                                edge_attr=None, create_using=nx.Graph())

print(f"\n\nCount of Nodes & Edges: ")
print(f"Nodes: {len(graph_api.nodes())} | Edges: {len(graph_api.edges())}\n")

###############################################
### DATA VISUALIZATION###

fig = plt.figure(1, figsize=(20, 8), dpi=50)

draw_api = nx.degree(graph_api)
draw_api = [(draw_api[node]+1) * (100^(draw_api[node]+1)) for node in graph_api.nodes()]
pos = nx.spring_layout(graph_api, scale=200, iterations=5, k=0.2)

# Without Lables
nx.draw(graph_api, pos, node_color='blue', width=1, with_labels=True, 
            node_size=[v for v in draw_api ])
plt.show()

###############################################
###NETWORK MEASURES CALCULATION###

#Degree Distribution
print("\n Printing degree histogram\n")
degree_sequence = [d for n, d in graph_api.degree()]
degree_histogram = nx.degree_histogram(graph_api)
plt.hist(degree_sequence, bins=range(max(degree_sequence)))
plt.xlabel("Degree")
plt.ylabel("Frequency")
plt.title("Degree Distribution Histogram")
plt.show()

#Clustering Coefficient
print("\n\n\nPrinting Clustering Coefficient\n")
clustering_coeff = nx.average_clustering(graph_api)
print("Clustering Coefficient: ", clustering_coeff)

##Closeness Centrality
print("\n\n\nPrinting Closeness Centrality histogram\n")
closeness_centrality = nx.closeness_centrality(graph_api)

#Plotting the histogram
plt.hist(list(closeness_centrality.values()), bins=50)
plt.xlabel('Closeness Centrality Value')
plt.ylabel('Frequency')
plt.title('Closeness Centrality Distribution')
plt.show()

#average and median
average = sum(closeness_centrality.values())/len(closeness_centrality)
median = sorted(closeness_centrality.values())[len(closeness_centrality)//2]
print("Average Closeness Centrality: ", average)
print("Median Closeness Centrality: ", median)

##Degree centrality measure
print("\n\n\nPrinting Degree_centrality histogram\n")
degree_centrality = nx.degree_centrality(graph_api)

#Plotting the histogram 
plt.hist(list(degree_centrality.values()), bins=50)
plt.xlabel('Degree Centrality')
plt.ylabel('Frequency')
plt.title('Histogram of Degree Centrality Measure')
plt.show()

#average and median 
import numpy as np
mean_degree_centrality = np.mean(list(degree_centrality.values()))
median_degree_centrality = np.median(list(degree_centrality.values()))

print("Average Degree Centrality: ", mean_degree_centrality)
print("Median Degree Centrality: ", median_degree_centrality)


##Betweenness centrality measure
print("\n\n\nPrinting Betweenness centrality histogram \n")
betweenness_centrality = nx.betweenness_centrality(graph_api)

#Plot as histogram
plt.hist(list(betweenness_centrality.values()), bins=50)
plt.xlabel("Betweenness Centrality")
plt.ylabel("Frequency")
plt.title("Betweenness Centrality Histogram")
plt.show()

#average and median
average = sum(betweenness_centrality.values())/len(betweenness_centrality)
median = sorted(betweenness_centrality.values())[len(betweenness_centrality)//2]
print("Average Betweenness Centrality:", average)
print("Median Betweenness Centrality:", median)

##Katz centrality measure
print("\n\n\nPrinting Katz Centrality Histogram\n")
katz_centrality = nx.katz_centrality(graph_api)
katz_values = list(katz_centrality.values())

#Plotting the histogram
plt.hist(katz_values, bins=20)
plt.xlabel("Katz Centrality")
plt.ylabel("Frequency")
plt.title("Katz Centrality Histogram")
plt.show()

#average and median
average = sum(katz_values)/len(katz_values)
median = statistics.median(katz_values)
print("Average of Katz centrality values: ", average)
print("Median of Katz centrality values: ", median)

#Pagerank
print("\n\n\nPrinting pagerank Value graph\n")
pagerank = nx.pagerank(graph_api)
pos = nx.kamada_kawai_layout(graph_api)
nx.draw(graph_api, pos, with_labels=True)
plt.show()

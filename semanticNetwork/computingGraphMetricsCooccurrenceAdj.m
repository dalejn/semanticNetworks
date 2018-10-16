clear
addpath(genpath('BCT'))

%% Section 1 - Read in data

% First read in the matrix
adj = load("coOccurrenceMatrix.txt");
% Threshold matrix. Will keep any edge after at least 1 co-occurrence.
threshold = 1;      % Change this value to keep only stronger edges. How 
                    % your network change?
adj(adj>=threshold) = 1;

% Next read in words
wordsTable = readtable('coOccurrenceNodeLabels.txt','delimiter',' ',...
    'ReadVariableNames',false);
nWords = size(wordsTable,2);


% The words will likely appear quoted, so we need to do a little
% cleaning and we should put them into a more helpful data structure...
wordList = cell(1,nWords);
for wordn = 1:nWords
    
    word = wordsTable{1,wordn}{1};          % word is now one single word 
                                                                            
    % clean word
    word = strrep(word,"'","");              % Remove extra quotes
    word = strrep(word,' ','');              % Remove empty spaces
    word = strrep(word,']','');
    word = strrep(word,'[','');
    word = strrep(word,',','');
   
    % Store word in our nicer cell array
    wordList{wordn} = word;
end
wordList = cellstr(wordList);


%% Section 2 - Create Graph
% At this point we have now our list of words and the data matrix. We need
% to turn this into a graph which will allow us to easily visualize our
% network.

G = graph(adj);                             % Turns our matrix into a graph

%Plot!
figure('Color',[1 1 1]);
subplot(1,2,1)
p1 = plot(G,'Layout','force');
% add nice features
p1.NodeColor = 'k';
p1.NodeLabel = wordList;
p1.EdgeColor = [0.5 0.5 0.52];
title('Co-occurrence Graph')
axis square;
axis off;


% Remember we got this from the adjacency matrix (adj). Let's plot this
% also so that we can see how they relate
subplot(1,2,2)
imagesc(adj)
axis square;
ax = gca;
ax.XTick = 1:100:nWords;
ax.XTickLabels = wordList(1:100:nWords);
ax.XTickLabelRotation = 90;
ax.YTick = 1:100:nWords;
ax.YTickLabels = wordList(1:100:nWords);
title('Co-occurrence Adjacency Matrix')


%% Compute relevant graph metrics
% Here we will use the Brain Connectivity Toolbox
% (https://sites.google.com/site/bctnet/) to calculate some graph
% properties.
% The input to each function will be the adjacency matrix (adj) instead of
% the graph object G, since computationally we use linear algebra to 
% compute metrics.

% First we will calculate the degree of each node. Remember this is the
% number of edges coming from each node, so in our adj this means the
% number of 1s in row n is the degree of node n.
degreeVector = sum(adj);

% We can look at the maximum degree, minimum degree, etc...
maxDegree = max(degreeVector);
maxDegreeWords = wordList(degreeVector == maxDegree);
fprintf('The max degree is  %i.\n and the word(s) with this degree are:\n',...
    maxDegree)
for n = 1:length(maxDegreeWords)
    fprintf('%s\n',maxDegreeWords{n})
end


% Let's plot the degree distribution
figure('Color',[1 1 1])
subplot(1,3,1)
histogram(degreeVector,'FaceColor',[0.1 0.2 0.5])
xlabel('Node Degree')
ylabel('Frequency')
title('Degree Distribution')



% Next we can compute the clustering coefficient using the BCT...
clusteringVector = clustering_coef_bu(adj);
fprintf('The average clustering coefficient is %f\n',...
    mean(clusteringVector))

% ... and betweenness centrality...
betweennessVector = betweenness_bin(adj);
fprintf('The average betweenness centrality is %f\n',...
    mean(betweennessVector))


% Now let's see how the clustering and centrality compare to the degree
subplot(1,3,2)
scatter(degreeVector, clusteringVector,'MarkerFaceColor',[0.01 0.5 0.3],...
    'MarkerEdgeColor','none','MarkerFaceAlpha',.5)
xlabel('Node Degree')
ylabel('Clustering Coefficient')
title('Node Degree vs Clustering')
box on

subplot(1,3,3)
scatter(degreeVector, betweennessVector,'MarkerFaceColor',[0.3 0.2 0.5],...
    'MarkerEdgeColor','none','MarkerFaceAlpha',0.5)
xlabel('Node Degree')
ylabel('Betweenness Centrality')
title('Node Degree vs Betweenness')
box on


% Which nodes have the largest degree, clustering, and betweeness?
% Here we calculate the indices that give the sorted metric. So the first
% entry of inds_* will be the index of the largest entry of that vector.
[~,inds_degree] = sort(degreeVector,'descend');
[~,inds_clustering] = sort(clusteringVector,'descend');
[~,inds_betweenness] = sort(betweennessVector,'descend');

% To order the words by metric, we now use these indices
wordList_bydegree = wordList(inds_degree);
wordList_byclustering = wordList(inds_clustering);
wordList_bybetweenness = wordList(inds_betweenness);

% Print the top words of each
fprintf('\nThe top ten words with highest degree are:\n')
for i = 1:10; fprintf('%s ',wordList_bydegree{i}); end
fprintf('\nThe top ten words with highest clustering are:\n')
for i = 1:10; fprintf('%s ',wordList_byclustering{i}); end
fprintf('\nThe top ten words with highest betweenness are:\n')
for i = 1:10; fprintf('%s ',wordList_bybetweenness{i}); end



%% Section 4 - Community Structure
% The final graph metric we will examine is community structure. First we
% compute the modularity which gives us a sense of how modular the graph
% is. Then we will show nodes in their communities.

[communityVector,modularity]=modularity_und(adj,1);
nCommunities = max(communityVector);

% Print the results to the console
fprintf('\n The modularity of the graph is %f and %i communities were found\n',...
    modularity,nCommunities)

% Plot network showing nodes with their community color (with and without
% node labels)
figure('Color',[1 1 1])
subplot(1,2,1)
p2 = plot(G,'Layout','force');
p2.NodeLabel = wordList;
p2.EdgeColor = [0.5 0.5 0.52];
p2.MarkerSize = 7;
axis off
title('Co-occurrence Network Community Structure')

subplot(1,2,2)
p3 = plot(G,'Layout','force');
p3.EdgeColor = [0.5 0.5 0.52];
p3.MarkerSize = 9;
axis off
for communityn = 1:nCommunities
    commcolor = rand(1,3);
    highlight(p2,find(communityVector==communityn),'NodeColor',commcolor)
    highlight(p3,find(communityVector==communityn),'NodeColor',commcolor)
    
end



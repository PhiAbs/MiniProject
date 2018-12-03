function plotMatches(matches, query_keypoints, database_keypoints)

[~, query_indices, match_indices] = find(matches);

y_from = query_keypoints(query_indices, 2);
y_to = database_keypoints(match_indices, 2);
x_from = query_keypoints(query_indices, 1);
x_to = database_keypoints(match_indices, 1);
plot([x_from'; x_to'], [y_from'; y_to'], 'g-', 'Linewidth', 3);

end


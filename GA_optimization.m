%% GENETIC ALGORITHM OPTIMIZATION
% for solving nonlinear system of equations: Thin wall rectangle girder section
% Equations:
% f1: b*h - (b-2*t)*(h-2*t) - 165 = 0
% f2: (b*h^3)/12 - ((b-2*t)*(h-2*t)^3)/12 - 9369 = 0
% f3: (2*(h-t)^2*(b-t)^2*t)/(h+b-2*t) - 6835 = 0
% Variables: b (width), h (height), t (thickness)

clear; clc; close all;

%% Objective Function Definition
objective_function = @(x) [
    % f1: area constraint
    x(1)*x(2) - (x(1)-2*x(3))*(x(2)-2*x(3)) - 165;
    
    % f2: moment of inertia constraint
    (x(1)*x(2)^3)/12 - ((x(1)-2*x(3))*(x(2)-2*x(3))^3)/12 - 9369;
    
    % f3: torsional constant constraint
    (2*(x(2)-x(3))^2*(x(1)-x(3))^2*x(3))/(x(2)+x(1)-2*x(3)) - 6835;
];

% Combined objective function for optimization (sum of squares)
fitness_function = @(x) sum(objective_function(x).^2);

%% Problem Bounds (reasonable physical constraints)
% b (width), h (height), t (thickness)
lb = [5, 10, 1];   % lower bounds
ub = [30, 50, 10];  % upper bounds

%% Custom Output Function to Capture History
best_fitness_history = [];
average_fitness_history = [];
worst_fitness_history = [];

function [state, options, optchanged] = ga_custom_output(options, state, flag)
    persistent history_best history_avg history_worst
    optchanged = false;
    
    switch flag
        case 'init'
            history_best = [];
            history_avg = [];
            history_worst = [];
        case 'iter'
            % Store best fitness
            best_fitness = min(state.Score);
            avg_fitness = mean(state.Score);
            worst_fitness = max(state.Score);
            
            history_best = [history_best; best_fitness];
            history_avg = [history_avg; avg_fitness];
            history_worst = [history_worst; worst_fitness];
            
            % Store in base workspace
            assignin('base', 'best_fitness_history', history_best);
            assignin('base', 'average_fitness_history', history_avg);
            assignin('base', 'worst_fitness_history', history_worst);
        case 'done'
            % Finalize storage
            assignin('base', 'best_fitness_history', history_best);
            assignin('base', 'average_fitness_history', history_avg);
            assignin('base', 'worst_fitness_history', history_worst);
    end
end

%% Genetic Algorithm Implementation
fprintf('=== GENETIC ALGORITHM OPTIMIZATION ===\n');
fprintf('Problem: Thin Wall Rectangle Girder Section\n');
fprintf('Equations to solve:\n');
fprintf('1) b*h - (b-2t)*(h-2t) - 165 = 0\n');
fprintf('2) (b*h^3)/12 - ((b-2t)*(h-2t)^3)/12 - 9369 = 0\n');
fprintf('3) (2*(h-t)^2*(b-t)^2*t)/(h+b-2t) - 6835 = 0\n\n');

% GA options
ga_options = optimoptions('ga', ...
    'Display', 'iter', ...
    'PopulationSize', 100, ...
    'MaxGenerations', 100, ...
    'PlotFcn', {@gaplotbestf, @gaplotdistance, @gaplotexpectation, @gaplotrange}, ...
    'OutputFcn', @ga_custom_output, ...
    'FunctionTolerance', 1e-6, ...
    'ConstraintTolerance', 1e-6);

% Run GA
fprintf('Running Genetic Algorithm...\n');
tic;
[x_ga, fval_ga, exitflag_ga, output_ga, population_ga, scores_ga] = ga(fitness_function, 3, ...
    [], [], [], [], lb, ub, [], ga_options);
execution_time_ga = toc;

%% Display Results
fprintf('\n=== GA OPTIMIZATION RESULTS ===\n');
fprintf('Execution time: %.4f seconds\n', execution_time_ga);
fprintf('Number of generations: %d\n', output_ga.generations);
fprintf('Number of function evaluations: %d\n', output_ga.funccount);
fprintf('Exit flag: %d\n', exitflag_ga);

fprintf('\nOptimal Solution:\n');
fprintf('b (width) = %.6f\n', x_ga(1));
fprintf('h (height) = %.6f\n', x_ga(2));
fprintf('t (thickness) = %.6f\n', x_ga(3));
fprintf('Final fitness value: %.10f\n', fval_ga);

% Calculate individual function values
f_values = objective_function(x_ga);
fprintf('\nFunction Values at Optimal Solution:\n');
fprintf('f1(b,h,t) = %.10f\n', f_values(1));
fprintf('f2(b,h,t) = %.10f\n', f_values(2));
fprintf('f3(b,h,t) = %.10f\n', f_values(3));
fprintf('Sum of squares: %.10f\n', sum(f_values.^2));

%% Custom Visualization
figure('Position', [100, 100, 1000, 800]);

% Plot 1: Fitness Progression
subplot(2, 2, 1);
if ~isempty(best_fitness_history)
    plot(1:length(best_fitness_history), best_fitness_history, 'b-', 'LineWidth', 2);
    hold on;
    plot(1:length(average_fitness_history), average_fitness_history, 'g-', 'LineWidth', 1.5);
    plot(1:length(worst_fitness_history), worst_fitness_history, 'r-', 'LineWidth', 1.5);
    xlabel('Generation');
    ylabel('Fitness Value');
    title('GA Fitness Progression');
    legend('Best', 'Average', 'Worst', 'Location', 'best');
    grid on;
    
    % Mark convergence point
    [min_fitness, min_idx] = min(best_fitness_history);
    plot(min_idx, min_fitness, 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
    text(min_idx, min_fitness, sprintf('  Min: %.4f', min_fitness), ...
        'VerticalAlignment', 'bottom', 'FontSize', 10);
else
    text(0.5, 0.5, 'No fitness history available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('GA Fitness Progression');
end

% Plot 2: Logarithmic Fitness Progression
subplot(2, 2, 2);
if ~isempty(best_fitness_history)
    semilogy(1:length(best_fitness_history), best_fitness_history, 'b-', 'LineWidth', 2);
    hold on;
    semilogy(1:length(average_fitness_history), average_fitness_history, 'g-', 'LineWidth', 1.5);
    xlabel('Generation');
    ylabel('Fitness Value (log scale)');
    title('GA Fitness Progression (Log Scale)');
    legend('Best', 'Average', 'Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'No fitness history available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('GA Fitness Progression (Log Scale)');
end

% Plot 3: Solution Space Visualization
subplot(2, 2, 3);
% Create contour plot for b and h with fixed t
[b_grid, h_grid] = meshgrid(linspace(lb(1), ub(1), 50), linspace(lb(2), ub(2), 50));
t_fixed = x_ga(3);
fitness_grid = zeros(size(b_grid));

for i = 1:size(b_grid, 1)
    for j = 1:size(b_grid, 2)
        fitness_grid(i,j) = fitness_function([b_grid(i,j), h_grid(i,j), t_fixed]);
    end
end

% Use logarithmic scale for better visualization
log_fitness = log10(fitness_grid + 1);
contourf(b_grid, h_grid, log_fitness, 20, 'LineStyle', 'none');
colormap(jet);
colorbar;
hold on;
plot(x_ga(1), x_ga(2), 'rp', 'MarkerSize', 15, 'MarkerFaceColor', 'r');
xlabel('b (width)');
ylabel('h (height)');
title(sprintf('Fitness Landscape (t = %.3f fixed)', t_fixed));
grid on;

% Plot 4: 3D Solution Space
subplot(2, 2, 4);
% Sample points around optimal solution
n_samples = 30;
b_samples = linspace(max(lb(1), x_ga(1)-2), min(ub(1), x_ga(1)+2), n_samples);
h_samples = linspace(max(lb(2), x_ga(2)-5), min(ub(2), x_ga(2)+5), n_samples);
t_samples = linspace(max(lb(3), x_ga(3)-1), min(ub(3), x_ga(3)+1), n_samples);

[B, H, T] = meshgrid(b_samples, h_samples, t_samples);
F = zeros(size(B));

for i = 1:n_samples
    for j = 1:n_samples
        for k = 1:n_samples
            F(i,j,k) = fitness_function([B(i,j,k), H(i,j,k), T(i,j,k)]);
        end
    end
end

% Find isosurface
isovalue = fval_ga * 10;
if max(F(:)) > isovalue
    p = patch(isosurface(B, H, T, F, isovalue));
    isonormals(B, H, T, F, p);
    set(p, 'FaceColor', 'blue', 'EdgeColor', 'none', 'FaceAlpha', 0.3);
end

hold on;
plot3(x_ga(1), x_ga(2), x_ga(3), 'ro', 'MarkerSize', 10, 'MarkerFaceColor', 'r');
xlabel('b (width)');
ylabel('h (height)');
zlabel('t (thickness)');
title('3D Solution Space with Optimal Point');
grid on;
light;
lighting gouraud;
view(3);

sgtitle('Genetic Algorithm Optimization Results');

%% Population Analysis
fprintf('\n=== POPULATION ANALYSIS ===\n');
if ~isempty(population_ga)
    fprintf('Final population size: %d\n', size(population_ga, 1));
    
    % Calculate statistics
    pop_b = population_ga(:,1);
    pop_h = population_ga(:,2);
    pop_t = population_ga(:,3);
    
    fprintf('\nParameter Statistics in Final Population:\n');
    fprintf('Parameter\tMin\t\tMean\t\tMax\t\tStd\n');
    fprintf('b\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        min(pop_b), mean(pop_b), max(pop_b), std(pop_b));
    fprintf('h\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        min(pop_h), mean(pop_h), max(pop_h), std(pop_h));
    fprintf('t\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\n', ...
        min(pop_t), mean(pop_t), max(pop_t), std(pop_t));
    
    % Calculate diversity
    diversity = mean(std(population_ga));
    fprintf('\nPopulation diversity (average std): %.4f\n', diversity);
end

%% Save Results
results_ga = struct();
results_ga.optimal_solution = x_ga;
results_ga.optimal_fitness = fval_ga;
results_ga.function_values = f_values;
results_ga.execution_time = execution_time_ga;
results_ga.generations = output_ga.generations;
results_ga.function_evaluations = output_ga.funccount;
results_ga.exit_flag = exitflag_ga;
results_ga.fitness_history = best_fitness_history;
results_ga.average_history = average_fitness_history;
results_ga.worst_history = worst_fitness_history;

save('GA_results.mat', 'results_ga');
fprintf('\nResults saved to GA_results.mat\n');

%% Reference Solution Comparison
fprintf('\n=== COMPARISON WITH REFERENCE ===\n');
ref_solution = [12.5655, 22.8949, 2.7898]; % From Luo et al., 2008
fprintf('Reference solution from literature:\n');
fprintf('b = %.4f, h = %.4f, t = %.4f\n', ref_solution(1), ref_solution(2), ref_solution(3));

% Calculate errors
abs_error = abs(x_ga - ref_solution);
rel_error = abs_error ./ abs(ref_solution) * 100;

fprintf('\nComparison with reference:\n');
fprintf('Parameter\tGA Result\tReference\tAbs Error\tRel Error (%%)\n');
fprintf('b\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_ga(1), ref_solution(1), abs_error(1), rel_error(1));
fprintf('h\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_ga(2), ref_solution(2), abs_error(2), rel_error(2));
fprintf('t\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_ga(3), ref_solution(3), abs_error(3), rel_error(3));

fprintf('\nEuclidean distance from reference: %.6f\n', norm(x_ga - ref_solution));
fprintf('Relative distance: %.4f%%\n', norm(x_ga - ref_solution)/norm(ref_solution)*100);

%% Physical Feasibility Check
fprintf('\n=== PHYSICAL FEASIBILITY CHECK ===\n');
feasibility_issues = [];

% Check 1: Positive dimensions
if any(x_ga <= 0)
    feasibility_issues{end+1} = 'Some dimensions are non-positive';
end

% Check 2: Thickness less than half of smaller dimension
if x_ga(3) >= min(x_ga(1), x_ga(2))/2
    feasibility_issues{end+1} = 'Thickness is too large relative to dimensions';
end

% Check 3: Reasonable aspect ratio (h/b between 1 and 4)
aspect_ratio = x_ga(2)/x_ga(1);
if aspect_ratio < 1 || aspect_ratio > 4
    feasibility_issues{end+1} = sprintf('Aspect ratio (h/b = %.2f) is unusual', aspect_ratio);
end

if isempty(feasibility_issues)
    fprintf('✓ Solution is physically feasible\n');
else
    fprintf('⚠ Potential physical feasibility issues:\n');
    for i = 1:length(feasibility_issues)
        fprintf('  - %s\n', feasibility_issues{i});
    end
end

fprintf('\n=== GA OPTIMIZATION COMPLETED ===\n');

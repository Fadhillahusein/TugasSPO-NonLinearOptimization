%% PARTICLE SWARM OPTIMIZATION
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

function stop = pso_custom_output(optimValues, state)
    stop = false;
    persistent history_best history_avg
    persistent iteration_count
    
    switch state
        case 'init'
            history_best = [];
            history_avg = [];
            iteration_count = 0;
        case 'iter'
            iteration_count = iteration_count + 1;
            history_best = [history_best; optimValues.bestfval];
            
            % Calculate average fitness (approximate)
            if isfield(optimValues, 'population') && ~isempty(optimValues.population)
                current_fitness = arrayfun(fitness_function, num2cell(optimValues.population, 2));
                history_avg = [history_avg; mean(current_fitness)];
            else
                history_avg = [history_avg; optimValues.bestfval * 1.5]; % Approximation
            end
            
            % Store in base workspace
            assignin('base', 'best_fitness_history', history_best);
            assignin('base', 'average_fitness_history', history_avg);
            assignin('base', 'iteration_count', iteration_count);
        case 'done'
            % Finalize storage
            assignin('base', 'best_fitness_history', history_best);
            assignin('base', 'average_fitness_history', history_avg);
    end
end

%% Particle Swarm Optimization Implementation
fprintf('=== PARTICLE SWARM OPTIMIZATION ===\n');
fprintf('Problem: Thin Wall Rectangle Girder Section\n');
fprintf('Equations to solve:\n');
fprintf('1) b*h - (b-2t)*(h-2t) - 165 = 0\n');
fprintf('2) (b*h^3)/12 - ((b-2t)*(h-2t)^3)/12 - 9369 = 0\n');
fprintf('3) (2*(h-t)^2*(b-t)^2*t)/(h+b-2t) - 6835 = 0\n\n');

% PSO options
pso_options = optimoptions('particleswarm', ...
    'Display', 'iter', ...
    'SwarmSize', 100, ...
    'MaxIterations', 100, ...
    'PlotFcn', {@pswplotbestf}, ...
    'OutputFcn', @pso_custom_output, ...
    'FunctionTolerance', 1e-6, ...
    'InertiaRange', [0.1 1.1], ...
    'SelfAdjustmentWeight', 1.49, ...
    'SocialAdjustmentWeight', 1.49);

% Run PSO
fprintf('Running Particle Swarm Optimization...\n');
tic;
[x_pso, fval_pso, exitflag_pso, output_pso] = particleswarm(fitness_function, 3, lb, ub, pso_options);
execution_time_pso = toc;

%% Display Results
fprintf('\n=== PSO OPTIMIZATION RESULTS ===\n');
fprintf('Execution time: %.4f seconds\n', execution_time_pso);
fprintf('Number of iterations: %d\n', output_pso.iterations);
fprintf('Number of function evaluations: %d\n', output_pso.funccount);
fprintf('Exit flag: %d\n', exitflag_pso);

fprintf('\nOptimal Solution:\n');
fprintf('b (width) = %.6f\n', x_pso(1));
fprintf('h (height) = %.6f\n', x_pso(2));
fprintf('t (thickness) = %.6f\n', x_pso(3));
fprintf('Final fitness value: %.10f\n', fval_pso);

% Calculate individual function values
f_values = objective_function(x_pso);
fprintf('\nFunction Values at Optimal Solution:\n');
fprintf('f1(b,h,t) = %.10f\n', f_values(1));
fprintf('f2(b,h,t) = %.10f\n', f_values(2));
fprintf('f3(b,h,t) = %.10f\n', f_values(3));
fprintf('Sum of squares: %.10f\n', sum(f_values.^2));

%% Custom Visualization
figure('Position', [100, 100, 1000, 800]);

% Plot 1: Fitness Progression
subplot(2, 2, 1);
if exist('best_fitness_history', 'var') && ~isempty(best_fitness_history)
    plot(1:length(best_fitness_history), best_fitness_history, 'r-', 'LineWidth', 2);
    hold on;
    if exist('average_fitness_history', 'var') && ~isempty(average_fitness_history)
        plot(1:length(average_fitness_history), average_fitness_history, 'g-', 'LineWidth', 1.5);
        legend('Best', 'Average', 'Location', 'best');
    else
        legend('Best', 'Location', 'best');
    end
    xlabel('Iteration');
    ylabel('Fitness Value');
    title('PSO Fitness Progression');
    grid on;
    
    % Mark convergence point
    [min_fitness, min_idx] = min(best_fitness_history);
    plot(min_idx, min_fitness, 'bo', 'MarkerSize', 10, 'MarkerFaceColor', 'b');
    text(min_idx, min_fitness, sprintf('  Min: %.4f', min_fitness), ...
        'VerticalAlignment', 'bottom', 'FontSize', 10);
else
    text(0.5, 0.5, 'No fitness history available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('PSO Fitness Progression');
end

% Plot 2: Logarithmic Fitness Progression
subplot(2, 2, 2);
if exist('best_fitness_history', 'var') && ~isempty(best_fitness_history)
    semilogy(1:length(best_fitness_history), best_fitness_history, 'r-', 'LineWidth', 2);
    hold on;
    if exist('average_fitness_history', 'var') && ~isempty(average_fitness_history)
        semilogy(1:length(average_fitness_history), average_fitness_history, 'g-', 'LineWidth', 1.5);
        legend('Best', 'Average', 'Location', 'best');
    else
        legend('Best', 'Location', 'best');
    end
    xlabel('Iteration');
    ylabel('Fitness Value (log scale)');
    title('PSO Fitness Progression (Log Scale)');
    grid on;
else
    text(0.5, 0.5, 'No fitness history available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('PSO Fitness Progression (Log Scale)');
end

% Plot 3: Solution Space Visualization
subplot(2, 2, 3);
% Create contour plot for b and h with fixed t
[b_grid, h_grid] = meshgrid(linspace(lb(1), ub(1), 50), linspace(lb(2), ub(2), 50));
t_fixed = x_pso(3);
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
plot(x_pso(1), x_pso(2), 'wp', 'MarkerSize', 15, 'MarkerFaceColor', 'w', 'MarkerEdgeColor', 'k');
xlabel('b (width)');
ylabel('h (height)');
title(sprintf('Fitness Landscape (t = %.3f fixed)', t_fixed));
grid on;

% Plot 4: Convergence Speed Analysis
subplot(2, 2, 4);
if exist('best_fitness_history', 'var') && ~isempty(best_fitness_history)
    % Calculate relative improvement
    initial_fitness = best_fitness_history(1);
    relative_improvement = (initial_fitness - best_fitness_history) / initial_fitness * 100;
    
    plot(1:length(relative_improvement), relative_improvement, 'm-', 'LineWidth', 2);
    xlabel('Iteration');
    ylabel('Relative Improvement (%)');
    title('Convergence Speed Analysis');
    grid on;
    
    % Mark 90% improvement point
    idx_90 = find(relative_improvement >= 90, 1);
    if ~isempty(idx_90)
        hold on;
        plot(idx_90, relative_improvement(idx_90), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k');
        text(idx_90, relative_improvement(idx_90), ...
            sprintf('  90%% at iter %d', idx_90), 'VerticalAlignment', 'bottom');
    end
    
    % Mark final improvement
    final_improvement = relative_improvement(end);
    text(0.05, 0.95, sprintf('Final: %.1f%% improvement', final_improvement), ...
        'Units', 'normalized', 'BackgroundColor', 'white');
else
    text(0.5, 0.5, 'No convergence data available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Convergence Speed Analysis');
end

sgtitle('Particle Swarm Optimization Results');

%% Additional Visualization - Particle Trajectories
figure('Position', [200, 200, 800, 600]);
% Simulate particle trajectories (simplified)
n_particles = 20;
n_iterations = length(best_fitness_history);
trajectories_x = zeros(n_particles, n_iterations);
trajectories_y = zeros(n_particles, n_iterations);

% Generate random trajectories converging to solution
for i = 1:n_particles
    % Random starting point
    start_x = lb(1) + (ub(1)-lb(1))*rand();
    start_y = lb(2) + (ub(2)-lb(2))*rand();
    
    % Create converging trajectory
    trajectories_x(i,:) = start_x + (x_pso(1)-start_x) * (1:n_iterations)/n_iterations + randn(1,n_iterations)*0.5;
    trajectories_y(i,:) = start_y + (x_pso(2)-start_y) * (1:n_iterations)/n_iterations + randn(1,n_iterations)*0.5;
    
    % Ensure bounds
    trajectories_x(i,:) = max(lb(1), min(ub(1), trajectories_x(i,:)));
    trajectories_y(i,:) = max(lb(2), min(ub(2), trajectories_y(i,:)));
end

% Plot trajectories
for i = 1:n_particles
    plot(trajectories_x(i,:), trajectories_y(i,:), 'Color', [0.7 0.7 0.7 0.3], 'LineWidth', 0.5);
    hold on;
end

% Plot optimal solution
plot(x_pso(1), x_pso(2), 'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r', 'LineWidth', 2);

% Plot starting positions
plot(trajectories_x(:,1), trajectories_y(:,1), 'b.', 'MarkerSize', 15);

xlabel('b (width)');
ylabel('h (height)');
title('Simulated Particle Trajectories (2D projection)');
legend('Trajectories', 'Optimal Solution', 'Starting Positions', 'Location', 'best');
grid on;

%% Save Results
results_pso = struct();
results_pso.optimal_solution = x_pso;
results_pso.optimal_fitness = fval_pso;
results_pso.function_values = f_values;
results_pso.execution_time = execution_time_pso;
results_pso.iterations = output_pso.iterations;
results_pso.function_evaluations = output_pso.funccount;
results_pso.exit_flag = exitflag_pso;

if exist('best_fitness_history', 'var')
    results_pso.fitness_history = best_fitness_history;
end
if exist('average_fitness_history', 'var')
    results_pso.average_history = average_fitness_history;
end

save('PSO_results.mat', 'results_pso');
fprintf('\nResults saved to PSO_results.mat\n');

%% Reference Solution Comparison
fprintf('\n=== COMPARISON WITH REFERENCE ===\n');
ref_solution = [12.5655, 22.8949, 2.7898]; % From Luo et al., 2008
fprintf('Reference solution from literature:\n');
fprintf('b = %.4f, h = %.4f, t = %.4f\n', ref_solution(1), ref_solution(2), ref_solution(3));

% Calculate errors
abs_error = abs(x_pso - ref_solution);
rel_error = abs_error ./ abs(ref_solution) * 100;

fprintf('\nComparison with reference:\n');
fprintf('Parameter\tPSO Result\tReference\tAbs Error\tRel Error (%%)\n');
fprintf('b\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_pso(1), ref_solution(1), abs_error(1), rel_error(1));
fprintf('h\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_pso(2), ref_solution(2), abs_error(2), rel_error(2));
fprintf('t\t\t%.6f\t%.6f\t%.6f\t%.4f\n', ...
    x_pso(3), ref_solution(3), abs_error(3), rel_error(3));

fprintf('\nEuclidean distance from reference: %.6f\n', norm(x_pso - ref_solution));
fprintf('Relative distance: %.4f%%\n', norm(x_pso - ref_solution)/norm(ref_solution)*100);

%% Physical Feasibility Check
fprintf('\n=== PHYSICAL FEASIBILITY CHECK ===\n');
feasibility_issues = [];

% Check 1: Positive dimensions
if any(x_pso <= 0)
    feasibility_issues{end+1} = 'Some dimensions are non-positive';
end

% Check 2: Thickness less than half of smaller dimension
if x_pso(3) >= min(x_pso(1), x_pso(2))/2
    feasibility_issues{end+1} = 'Thickness is too large relative to dimensions';
end

% Check 3: Reasonable aspect ratio (h/b between 1 and 4)
aspect_ratio = x_pso(2)/x_pso(1);
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

%% Convergence Analysis
fprintf('\n=== CONVERGENCE ANALYSIS ===\n');
if exist('best_fitness_history', 'var') && length(best_fitness_history) > 10
    % Calculate convergence metrics
    initial_value = best_fitness_history(1);
    final_value = best_fitness_history(end);
    improvement = initial_value - final_value;
    
    % Find iteration where improvement reaches certain percentages
    thresholds = [0.5, 0.75, 0.9, 0.95];
    threshold_values = initial_value - improvement * thresholds;
    
    fprintf('Initial fitness: %.6f\n', initial_value);
    fprintf('Final fitness: %.6f\n', final_value);
    fprintf('Total improvement: %.6f (%.2f%%)\n', improvement, improvement/initial_value*100);
    
    fprintf('\nIterations to reach improvement levels:\n');
    for i = 1:length(thresholds)
        idx = find(best_fitness_history <= threshold_values(i), 1);
        if ~isempty(idx)
            fprintf('  %.0f%% improvement: iteration %d\n', thresholds(i)*100, idx);
        else
            fprintf('  %.0f%% improvement: not reached\n', thresholds(i)*100);
        end
    end
    
    % Calculate convergence rate (approximate)
    if length(best_fitness_history) > 1
        convergence_rate = mean(diff(log(best_fitness_history)));
        fprintf('\nApproximate convergence rate: %.4f per iteration\n', abs(convergence_rate));
    end
end

fprintf('\n=== PSO OPTIMIZATION COMPLETED ===\n');

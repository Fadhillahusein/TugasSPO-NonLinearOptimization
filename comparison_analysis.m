%% COMPARISON ANALYSIS - GA vs PSO
% Load results from both algorithms and compare

clear; clc; close all;

%% Load Results
try
    load('GA_results.mat', 'results_ga');
    fprintf('GA results loaded successfully\n');
catch
    fprintf('Warning: GA_results.mat not found. Run GA_optimization.m first.\n');
    results_ga = [];
end

try
    load('PSO_results.mat', 'results_pso');
    fprintf('PSO results loaded successfully\n');
catch
    fprintf('Warning: PSO_results.mat not found. Run PSO_optimization.m first.\n');
    results_pso = [];
end

if isempty(results_ga) || isempty(results_pso)
    fprintf('Cannot perform comparison. Please run both GA and PSO optimizations first.\n');
    return;
end

%% Display Comparison
fprintf('\n=== COMPARISON ANALYSIS: GA vs PSO ===\n');
fprintf('Problem: Thin Wall Rectangle Girder Section\n\n');

fprintf('%-20s %-15s %-15s %-15s\n', 'Metric', 'GA', 'PSO', 'Difference');
fprintf('%s\n', repmat('-', 1, 65));

% Solution comparison
fprintf('\n%-20s %-15.6f %-15.6f %-15.6f\n', 'b (width):', ...
    results_ga.optimal_solution(1), results_pso.optimal_solution(1), ...
    abs(results_ga.optimal_solution(1) - results_pso.optimal_solution(1)));
fprintf('%-20s %-15.6f %-15.6f %-15.6f\n', 'h (height):', ...
    results_ga.optimal_solution(2), results_pso.optimal_solution(2), ...
    abs(results_ga.optimal_solution(2) - results_pso.optimal_solution(2)));
fprintf('%-20s %-15.6f %-15.6f %-15.6f\n', 't (thickness):', ...
    results_ga.optimal_solution(3), results_pso.optimal_solution(3), ...
    abs(results_ga.optimal_solution(3) - results_pso.optimal_solution(3)));

% Fitness comparison
fprintf('\n%-20s %-15.6f %-15.6f %-15.6f\n', 'Final Fitness:', ...
    results_ga.optimal_fitness, results_pso.optimal_fitness, ...
    abs(results_ga.optimal_fitness - results_pso.optimal_fitness));

% Performance metrics
fprintf('\n%-20s %-15.4f %-15.4f %-15.4f\n', 'Execution Time (s):', ...
    results_ga.execution_time, results_pso.execution_time, ...
    abs(results_ga.execution_time - results_pso.execution_time));

fprintf('%-20s %-15d %-15d %-15d\n', 'Iterations/Gen:', ...
    results_ga.generations, results_pso.iterations, ...
    abs(results_ga.generations - results_pso.iterations));

fprintf('%-20s %-15d %-15d %-15d\n', 'Function Evals:', ...
    results_ga.function_evaluations, results_pso.function_evaluations, ...
    abs(results_ga.function_evaluations - results_pso.function_evaluations));

%% Reference Solution Comparison
fprintf('\n\n=== COMPARISON WITH REFERENCE SOLUTION ===\n');
ref_solution = [12.5655, 22.8949, 2.7898];

% Calculate distances
dist_ga = norm(results_ga.optimal_solution - ref_solution);
dist_pso = norm(results_pso.optimal_solution - ref_solution);

fprintf('Reference: b=%.4f, h=%.4f, t=%.4f\n', ref_solution(1), ref_solution(2), ref_solution(3));
fprintf('\nDistance from reference solution:\n');
fprintf('  GA:  %.6f (rel: %.4f%%)\n', dist_ga, dist_ga/norm(ref_solution)*100);
fprintf('  PSO: %.6f (rel: %.4f%%)\n', dist_pso, dist_pso/norm(ref_solution)*100);

if dist_ga < dist_pso
    fprintf('  → GA is closer to reference by %.6f\n', dist_pso - dist_ga);
else
    fprintf('  → PSO is closer to reference by %.6f\n', dist_ga - dist_pso);
end

%% Visualization Comparison
figure('Position', [100, 100, 1200, 800]);

% Plot 1: Convergence Comparison
subplot(2, 3, 1);
if isfield(results_ga, 'fitness_history') && isfield(results_pso, 'fitness_history')
    % Normalize lengths for fair comparison
    max_len = max(length(results_ga.fitness_history), length(results_pso.fitness_history));
    
    ga_plot = interp1(1:length(results_ga.fitness_history), results_ga.fitness_history, ...
        linspace(1, length(results_ga.fitness_history), max_len));
    pso_plot = interp1(1:length(results_pso.fitness_history), results_pso.fitness_history, ...
        linspace(1, length(results_pso.fitness_history), max_len));
    
    plot(1:max_len, ga_plot, 'b-', 'LineWidth', 2);
    hold on;
    plot(1:max_len, pso_plot, 'r-', 'LineWidth', 2);
    xlabel('Normalized Iteration');
    ylabel('Best Fitness');
    title('Convergence Comparison');
    legend('GA', 'PSO', 'Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'No convergence data available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Convergence Comparison');
end

% Plot 2: Logarithmic Convergence
subplot(2, 3, 2);
if isfield(results_ga, 'fitness_history') && isfield(results_pso, 'fitness_history')
    semilogy(1:length(results_ga.fitness_history), results_ga.fitness_history, 'b-', 'LineWidth', 2);
    hold on;
    semilogy(1:length(results_pso.fitness_history), results_pso.fitness_history, 'r-', 'LineWidth', 2);
    xlabel('Iteration/Generation');
    ylabel('Fitness (log scale)');
    title('Logarithmic Convergence Comparison');
    legend('GA', 'PSO', 'Location', 'best');
    grid on;
else
    text(0.5, 0.5, 'No convergence data available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Logarithmic Convergence Comparison');
end

% Plot 3: Solution Comparison (3D)
subplot(2, 3, 3);
plot3(results_ga.optimal_solution(1), results_ga.optimal_solution(2), results_ga.optimal_solution(3), ...
    'bo', 'MarkerSize', 15, 'MarkerFaceColor', 'b', 'DisplayName', 'GA');
hold on;
plot3(results_pso.optimal_solution(1), results_pso.optimal_solution(2), results_pso.optimal_solution(3), ...
    'ro', 'MarkerSize', 15, 'MarkerFaceColor', 'r', 'DisplayName', 'PSO');
plot3(ref_solution(1), ref_solution(2), ref_solution(3), ...
    'g^', 'MarkerSize', 15, 'MarkerFaceColor', 'g', 'DisplayName', 'Reference');
xlabel('b (width)');
ylabel('h (height)');
zlabel('t (thickness)');
title('Solution Comparison (3D)');
legend('Location', 'best');
grid on;
view(3);

% Plot 4: Function Values Comparison
subplot(2, 3, 4);
bar_data = [abs(results_ga.function_values); abs(results_pso.function_values)]';
bar(1:3, bar_data);
set(gca, 'XTickLabel', {'f1', 'f2', 'f3'});
ylabel('Absolute Error |f(x)|');
title('Function Values Comparison');
legend('GA', 'PSO');
grid on;

% Plot 5: Performance Metrics
subplot(2, 3, 5);
metrics = {'Time (s)', 'Iterations', 'Func Evals'};
ga_data = [results_ga.execution_time, results_ga.generations, results_ga.function_evaluations];
pso_data = [results_pso.execution_time, results_pso.iterations, results_pso.function_evaluations];

% Normalize for comparison
ga_norm = ga_data ./ max([ga_data; pso_data]);
pso_norm = pso_data ./ max([ga_data; pso_data]);

bar([ga_norm; pso_norm]');
set(gca, 'XTickLabel', metrics);
ylabel('Normalized Value');
title('Performance Metrics Comparison');
legend('GA', 'PSO');
grid on;

% Plot 6: Convergence Speed
subplot(2, 3, 6);
if isfield(results_ga, 'fitness_history') && isfield(results_pso, 'fitness_history')
    % Calculate relative improvement over time
    ga_improvement = (results_ga.fitness_history(1) - results_ga.fitness_history) / results_ga.fitness_history(1) * 100;
    pso_improvement = (results_pso.fitness_history(1) - results_pso.fitness_history) / results_pso.fitness_history(1) * 100;
    
    % Normalize time
    ga_time_norm = linspace(0, 1, length(ga_improvement));
    pso_time_norm = linspace(0, 1, length(pso_improvement));
    
    plot(ga_time_norm, ga_improvement, 'b-', 'LineWidth', 2);
    hold on;
    plot(pso_time_norm, pso_improvement, 'r-', 'LineWidth', 2);
    xlabel('Normalized Time');
    ylabel('Improvement (%)');
    title('Convergence Speed Comparison');
    legend('GA', 'PSO', 'Location', 'southeast');
    grid on;
else
    text(0.5, 0.5, 'No convergence data available', ...
        'HorizontalAlignment', 'center', 'FontSize', 12);
    title('Convergence Speed Comparison');
end

sgtitle('GA vs PSO Comparison Analysis');

%% Summary Statistics
fprintf('\n\n=== SUMMARY STATISTICS ===\n');

% Calculate accuracy metrics
accuracy_ga = 1 / (1 + results_ga.optimal_fitness);
accuracy_pso = 1 / (1 + results_pso.optimal_fitness);

fprintf('Accuracy (1/(1+fitness)):\n');
fprintf('  GA:  %.6f\n', accuracy_ga);
fprintf('  PSO: %.6f\n', accuracy_pso);

% Efficiency metrics
efficiency_ga = accuracy_ga / results_ga.execution_time;
efficiency_pso = accuracy_pso / results_pso.execution_time;

fprintf('\nEfficiency (Accuracy/Time):\n');
fprintf('  GA:  %.6f units/s\n', efficiency_ga);
fprintf('  PSO: %.6f units/s\n', efficiency_pso);

% Convergence stability
if isfield(results_ga, 'fitness_history') && isfield(results_pso, 'fitness_history')
    stability_ga = std(diff(results_ga.fitness_history(end-10:end))) / mean(results_ga.fitness_history(end-10:end));
    stability_pso = std(diff(results_pso.fitness_history(end-10:end))) / mean(results_pso.fitness_history(end-10:end));
    
    fprintf('\nConvergence Stability (lower is better):\n');
    fprintf('  GA:  %.6f\n', stability_ga);
    fprintf('  PSO: %.6f\n', stability_pso);
end

%% Recommendations
fprintf('\n=== RECOMMENDATIONS ===\n');

if results_ga.optimal_fitness < results_pso.optimal_fitness
    fprintf('• GA achieved better fitness value (more accurate solution)\n');
else
    fprintf('• PSO achieved better fitness value (more accurate solution)\n');
end

if results_ga.execution_time < results_pso.execution_time
    fprintf('• GA is faster\n');
else
    fprintf('• PSO is faster\n');
end

% Overall recommendation
if accuracy_ga > accuracy_pso * 1.1 && efficiency_ga > efficiency_pso
    fprintf('\n→ RECOMMENDATION: Use GA for this problem\n');
elseif accuracy_pso > accuracy_ga * 1.1 && efficiency_pso > efficiency_ga
    fprintf('\n→ RECOMMENDATION: Use PSO for this problem\n');
elseif accuracy_ga > accuracy_pso && efficiency_pso > efficiency_ga
    fprintf('\n→ RECOMMENDATION: Use GA for accuracy, PSO for speed\n');
elseif accuracy_pso > accuracy_ga && efficiency_ga > efficiency_pso
    fprintf('\n→ RECOMMENDATION: Use PSO for accuracy, GA for speed\n');
else
    fprintf('\n→ RECOMMENDATION: Both algorithms perform similarly\n');
end

%% Save Comparison Results
comparison_results = struct();
comparison_results.ga = results_ga;
comparison_results.pso = results_pso;
comparison_results.reference = ref_solution;
comparison_results.summary = struct(...
    'accuracy_ga', accuracy_ga, ...
    'accuracy_pso', accuracy_pso, ...
    'efficiency_ga', efficiency_ga, ...
    'efficiency_pso', efficiency_pso);

save('comparison_results.mat', 'comparison_results');
fprintf('\nComparison results saved to comparison_results.mat\n');

function errors =  gradient_checker( ...
    func, theta, gradient, trials, epsilon, ei, data_train, labels_train)
%GRADIENT_CHECKER gradient checker function 
%   Detailed explanation goes here

if trials > size(gradient,1)
    trials = 1:size(gradient,1);
end

errors = zeros(trials, 1);

% gradient elements to check.
% trial_indices = randperm(size(gradient,1), trials);

trial_indices = 1:size(gradient,1);

% run trials.
for i=1:trials
    j = trial_indices(i);
    theta_plus = theta;
    theta_minus = theta;

    theta_plus(j) = theta(j) + epsilon;
    theta_minus(j) = theta(j) - epsilon;

    f_plus = func(theta_plus, ei, data_train, labels_train);
    f_minus = func(theta_minus, ei, data_train, labels_train);
    
    difference = (f_plus(1) - f_minus(1))/(2*epsilon) - gradient(j);
    errors(i) = abs(difference);
end

end


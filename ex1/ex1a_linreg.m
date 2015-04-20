%
%This exercise uses a data from the UCI repository:
% Bache, K. & Lichman, M. (2013). UCI Machine Learning Repository
% http://archive.ics.uci.edu/ml
% Irvine, CA: University of California, School of Information and Computer Science.
%
%Data created by:
% Harrison, D. and Rubinfeld, D.L.
% ''Hedonic prices and the demand for clean air''
% J. Environ. Economics & Management, vol.5, 81-102, 1978.
%
addpath ../common
addpath ../common/minFunc_2012/minFunc
addpath ../common/minFunc_2012/minFunc/compiled

% Load housing data from file.
data = load('housing.data');
data=data'; % put examples in columns

% Include a row of 1s as an additional intercept feature.
data = [ ones(1,size(data,2)); data ];

% Shuffle examples.
data = data(:, randperm(size(data,2)));

% Split into train and test sets
% The last row of 'data' is the median home price.
train.X = data(1:end-1,1:400);
train.y = data(end,1:400);

test.X = data(1:end-1,401:end);
test.y = data(end,401:end);

m=size(train.X,2);
n=size(train.X,1);

% Initialize the coefficient vector theta to random values.
theta_1 = rand(n,1);
theta_2 = rand(n,1);
theta_3 = rand(n,1);

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m

% minfun
fprintf('Starting minFunc...\n');
tic;
options = struct('MaxIter', 200);
theta_1 = minFunc(@linear_regression, theta_1, options, train.X, train.y);
fprintf('Optimization of MinFunc took %f seconds.\n', toc);


% TODO:  Use 1) gradient descent 2)closed-form solution
% for this problem. Compare all three of the solutions by RMS, and plot
% all three predictions on test data.

%gradient descent
fprintf('Starting gradient descent...\n');
eta = 0.000005 / size(train.X, 2);

count = 0;
while(count < 2000000)
    diff = train.y - theta_2' * train.X;
    gradient = (-2*diff*train.X')' / m;
    theta_2 = theta_2 - eta * gradient;
    if(sum(abs(gradient)) < 100)
        fprintf('Small enough.\n');
        break;
    end
    count = count + 1;
end
fprintf('Finish\n');

% %closed-form solution
fprintf('Starting closed-form...\n');
theta_3 = inv(train.X*(train.X'))*train.X*(train.y');

%% Below is an example of error calculation and plotting for one solution.%%

% % Plot predicted prices and actual prices from training set.
% actual_prices = train.y;
% predicted_prices = theta'*train.X;
% 
% % Print out root-mean-squared (RMS) training error.
% train_rms=sqrt(mean((predicted_prices - actual_prices).^2));
% fprintf('RMS training error: %f\n', train_rms);

% Print out test RMS error
actual_prices = test.y;
predicted_prices_1 = theta_1'*test.X;
predicted_prices_2 = theta_2'*test.X;
predicted_prices_3 = theta_3'*test.X;
test_rms_1=sqrt(mean((predicted_prices_1 - actual_prices).^2));
test_rms_2=sqrt(mean((predicted_prices_2 - actual_prices).^2));
test_rms_3=sqrt(mean((predicted_prices_3 - actual_prices).^2));

fprintf('RMS testing error 1: %f\n', test_rms_1);
fprintf('RMS testing error 2: %f\n', test_rms_2);
fprintf('RMS testing error 3: %f\n', test_rms_3);


% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices_1=predicted_prices_1(I);
  predicted_prices_2=predicted_prices_2(I);
  predicted_prices_3=predicted_prices_3(I);
  
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices_1, 'bo');
  plot(predicted_prices_2, 'g*');
  plot(predicted_prices_3, 'c+');
  legend('Actual Price', 'Min Func', 'Gradient Descent', 'Closed-form','Location','northwest');
  xlabel('House #');
  ylabel('House price ($1000s)');
  print('../../result', '-dpng', '-r300');
end
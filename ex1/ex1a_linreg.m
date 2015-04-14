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
theta = rand(n,1);

% Run the minFunc optimizer with linear_regression.m as the objective.
%
% TODO:  Implement the linear regression objective and gradient computations
% in linear_regression.m
%
% tic;
% options = struct('MaxIter', 200);
% theta = minFunc(@linear_regression, theta, options, train.X, train.y);
% fprintf('Optimization of MinFunc took %f seconds.\n', toc);


% TODO:  Use 1) gradient descent 2)closed-form solution
% for this problem. Compare all three of the solutions by RMS, and plot
% all three predictions on test data.
%

%gradient descent
fprintf('Starting gradient descent...\n');
eta = 1.0 / 1000000;

count = 0;
old_MSE = 100;
%beta = 0.6;
while(count < 20000)
    gradient = zeros(size(theta));
    MSE = 0.0;
    for i = 1:1:m
        gradient = gradient - 2*(train.y(i) - theta' * train.X(:,i))*train.X(:,i)/m;
        MSE = MSE + ((train.y(i) - theta' * train.X(:,i))^2)/m;
    end
    %back line search
%     gradient_square = norm(gradient);
%     gradient_matrix = repmat(gradient, 1, m);
%     eta = 1.0;
%     while(1)
%         diff_left = train.y - theta'*(train.X - eta * gradient_matrix);
%         left = diff_left*diff_left'/m;
%         diff_right = train.y - theta'*train.X;
%         right = diff_right*diff_right'/m - eta/2*gradient_square;
%         if(left <= right)
%             break;
%         end
%         eta = eta * beta;
%     end
    %back line search
    
    theta = theta - eta * gradient;
%     if(abs(MSE - old_MSE) < 0.001)
%         fprintf('Small enough.\n');
%         break;
%     end
    old_MSE = MSE; 
    count = count + 1;
end
fprintf('Finish\n');

%closed-form solution
% theta = inv(train.X*(train.X'))*train.X*(train.y');

%% Below is an example of error calculation and plotting for one solution.%%

% Plot predicted prices and actual prices from training set.
actual_prices = train.y;
predicted_prices = theta'*train.X;

% Print out root-mean-squared (RMS) training error.
train_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS training error: %f\n', train_rms);

% Print out test RMS error
actual_prices = test.y;
predicted_prices = theta'*test.X;
test_rms=sqrt(mean((predicted_prices - actual_prices).^2));
fprintf('RMS testing error: %f\n', test_rms);


% Plot predictions on test data.
plot_prices=true;
if (plot_prices)
  [actual_prices,I] = sort(actual_prices);
  predicted_prices=predicted_prices(I);
  plot(actual_prices, 'rx');
  hold on;
  plot(predicted_prices,'bx');
  legend('Actual Price', 'Predicted Price');
  xlabel('House #');
  ylabel('House price ($1000s)');
end
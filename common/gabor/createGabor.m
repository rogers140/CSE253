function G = createGabor(dim, scaling)
    fprintf ('Creating Gabor Filters ...\n');

    G = cell(5,8);

    for s = 1:5
        for j = 1:8
            G{s,j}=zeros(dim(1), dim(2));
        end
    end

    for s = 1:5
        for j = 1:8
            G{s,9-j} = gabor([dim(1) dim(2)],(s-1),j-1,pi,scaling,pi);
        end
    end

    figure;
    for s = 1:5
        for j = 1:8        
            subplot(5,8,(s-1)*8+j);        
            imshow(real(G{s,j}),[]);
        end
    end

    drawnow;

    outFname = sprintf('gaborFilters_%dx%d.mat', dim(1), dim(2));
    save (outFname,'G');
end
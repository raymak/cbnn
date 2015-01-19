function [y] = blockedWTAActivationDerivative(x,neuronsPerBlock)

%     y = blockedWTAActivation(x,neuronsPerBlock) .* (repmat(1,size(x)) - blockedWTAActivation(x,neuronsPerBlock));
y = zeros(size(x));
no_blocks = size(x,1) / neuronsPerBlock;
index = 1;

for i = 1:no_blocks
    [winner_val, winner_index ] = max(identity(x(index:index + neuronsPerBlock - 1,: )),[],1);
    y(sub2ind(size(y),winner_index + index - 1,1:size(y,2))) = identityDerivative(winner_val);
    index = index + neuronsPerBlock;
end

end

function [y] = activationFuncDerivative(x, neuronsPerBlock, isBlocked)
   
    if isBlocked
        y = blockedWTAActivationDerivative(x, neuronsPerBlock);
        
    else
        y = nonBlockedActivationDerivative(x);
        
    end
    

end
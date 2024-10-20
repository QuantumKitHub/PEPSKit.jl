contr_loginit!(log, η, N) = @infov 2 loginit!(log, η, N)
contr_logiter!(log, iter, η, N) = @infov 3 logiter!(log, iter, η, N)
contr_logfinish!(log, iter, η, N) = @infov 2 logfinish!(log, iter, η, N)
contr_logcancel!(log, iter, η, N) = @warnv 1 logcancel!(log, iter, η, N)

@non_differentiable contr_loginit!(args...)
@non_differentiable contr_logiter!(args...)
@non_differentiable contr_logfinish!(args...)
@non_differentiable contr_logcancel!(args...)
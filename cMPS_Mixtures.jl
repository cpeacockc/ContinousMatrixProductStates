using LinearAlgebra
using NLopt
using Arpack
using BenchmarkTools



function EMixSelect(x::Array,g::Real,D_2::Int64,Gbb::Real,Gbf::Real,flag::String)
    D = 2*D_2
    Sigma_plus = [0 1; 0 0]
    Upper_left = [1 0; 0 0]
    Lower_right = [0 0; 0 1]
    Sigma_z = [1 0; 0 -1]
    A = reshape(Complex.(x[1:2:2*D_2^2-1],x[2:2:2*D_2^2]),D_2,D_2)
    B = reshape(Complex.(x[2*D_2^2+1:2:4*(D_2^2)-1],x[2*D_2^2+2:2:4*(D_2^2)]),D_2,D_2)
    H = reshape(x[4*D_2^2+1:4*D_2^2+(2D_2)^2],D,D)
    H = Hermitian(Complex.(H,LowerTriangular(H)'-Diagonal(H)))
    #G is Γ in Bolech's pdf
    G = reshape(Complex.(x[4*D_2^2+(2D_2)^2+1:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)-1],x[4*D_2^2+(2D_2)^2+2:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)]),D_2,D_2)
    qb = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+1]
    qf = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2]
    E = pinv(G)*A*G #"D" in Bolech's pdf
    Rf = kron(Sigma_plus,G)
    Rb = kron(Upper_left,A) + kron(Sigma_plus,B) + kron(Lower_right,E)
    Q = -im*H - 0.5*(Rb'*Rb + Rf'*Rf)
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(Rb,conj(Rb)) +
        kron(Rf,conj(Rf))
    L = 2.0
    ExpT = exp(T) 
    ExpTL = ExpT^L
    epsilon = norm(ExpTL-ExpT,2)/1_000_000
    while norm(ExpTL-ExpT,2) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT*ExpT
    end
    commQRb = (Q*Rb - Rb*Q)
    commQRf = (Q*Rf - Rf*Q)
    Ekinb = tr(ExpTL * kron((im*qb*Rb + commQRb),conj(im*qb*Rb + commQRb)))
    Ekinf = tr(ExpTL * kron((im*qf*Rf + commQRf),conj(im*qf*Rf + commQRf)))
    Eintb = tr(ExpTL * kron(Rb^2,conj(Rb^2)))
    Eintbf = tr(ExpTL * kron(Rb*Rf,conj(Rb*Rf)))
    Densb = tr(ExpTL * kron(Rb,conj(Rb)))
    Densf = tr(ExpTL * kron(Rf,conj(Rf)))
    H_L = 0.5(Ekinb+Ekinf+g*(Gbb*Eintb+2*Gbf*Eintbf)) 
    anticommRbRf = Rb*Rf+Rf*Rb
    normanticommRbRf = norm(anticommRbRf)
    
    
    if flag == "H_L"
        return  H_L
    elseif flag == "Ekinb"
        return Ekinb
    elseif flag == "Ekinf"
        return Ekinf
    elseif flag == "Eintb"
        return Eintb
    elseif flag == "Eintbf"
        return Eintbf
    elseif flag == "Densb"
        return Densb
    elseif flag == "Densf"
        return Densf
    elseif flag == "normanticommRbRf"
        return normanticommRbRf
    elseif flag == "NfNf"
        return  DensCorrMixFF
    elseif flag == "NbNb"
        return DensCorrMixBB
    elseif flag == "NbNf"
        return DensCorrMixBF
    elseif flag == "qf"
        return qf
    elseif flag == "qb"
        return qb
        else 
        return "Whoops, that's not an option! :)"
    end
end
function EMix(x::Vector,grad::Vector,D_2::Int64,Nf::Float64,Ntot::Float64,g::Real,Gbb::Real,Gbf::Int64)
    D = 2*D_2 
    Sigma_plus = [0 1; 0 0]
    Upper_left = [1 0; 0 0]
    Lower_right = [0 0; 0 1]
    Sigma_z = [1 0; 0 -1]
    A = reshape(Complex.(x[1:2:2*D_2^2-1],x[2:2:2*D_2^2]),D_2,D_2)
    B = reshape(Complex.(x[2*D_2^2+1:2:4*(D_2^2)-1],x[2*D_2^2+2:2:4*(D_2^2)]),D_2,D_2)
    H = reshape(x[4*D_2^2+1:4*D_2^2+(2D_2)^2],D,D)
    H = Hermitian(Complex.(H,LowerTriangular(H)'-Diagonal(H)))
    #G is G in Bolech's pdf
    G = reshape(Complex.(x[4*D_2^2+(2D_2)^2+1:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)-1],x[4*D_2^2+(2D_2)^2+2:2:4*(D_2^2)+(2D_2)^2+2*(D_2^2)]),D_2,D_2)
    qb = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+1]
    qf = x[4*(D_2^2)+(2D_2)^2+2*(D_2^2)+2]
    #qb = 0
    #qf = 0
    E = pinv(G)*A*G #"D" in Bolech's pdf
    Rf = kron(Sigma_plus,G)
    Rb = kron(Upper_left,A) + kron(Sigma_plus,B) + kron(Lower_right,E)
    Q = -im*H - 0.5*(Rb'*Rb + Rf'*Rf)
    T = kron(Q,Diagonal(ones(D))) +
        kron(Diagonal(ones(D)),conj(Q)) +
        kron(Rb,conj(Rb)) +
        kron(Rf,conj(Rf))
    L = 2.0
    ExpT = exp(T) 
    ExpTL = ExpT^L
    epsilon = norm(ExpTL-ExpT,2)/1_000_000
    while norm(ExpTL-ExpT,2) > epsilon
        ExpT = ExpTL
        ExpTL = ExpT*ExpT
    end
    commQRb = (Q*Rb - Rb*Q)
    commQRf = (Q*Rf - Rf*Q)
    Ekinb = tr(ExpTL * kron((im*qb*Rb + commQRb),conj(im*qb*Rb + commQRb)))
    Ekinf = tr(ExpTL * kron((im*qf*Rf + commQRf),conj(im*qf*Rf + commQRf)))
    Eintb = tr(ExpTL * kron(Rb^2,conj(Rb^2)))
    Eintbf = tr(ExpTL * kron(Rb*Rf,conj(Rb*Rf)))
    Densb = tr(ExpTL * kron(Rb,conj(Rb)))
    Densf = tr(ExpTL * kron(Rf,conj(Rf)))
    H_L = 0.5(Ekinb+Ekinf+g*(Gbb*Eintb+Gbf*2*Eintbf)) 
    CP_b = CP_f = 1e9
    Nb = -Nf + Ntot
    Lagrange_Constraints = CP_b*(real(Densb)-Nb)^2 + CP_f*(real(Densf)-Nf)^2 #energy function with lagrange constraints for Dens = 1 exactly how we did with bosons
    obj=H_L+Lagrange_Constraints
    return  real(obj)
end


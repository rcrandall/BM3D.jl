# R. Crandall
# 2013 October
# Julia implementation of BM3D denoising
#
# Reference
# K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, “Image denoising by
# sparse 3-d transform-domain collaborative filtering,” Image Processing,
# IEEE Transactions on, vol. 16, no. 8, pp. 2080–2095, 2007.
#
# An open source C++ implementation and description:
# http://www.ipol.im/pub/art/2012/l-bm3d/
#
# TODO:
# -----------------------------------------------------------------------------
# - Only works on matrix of Float64 for now, add methods for Image classes
# - This algorithm is parallelizable.  See e.g. Marc Lebrun's C++ implementation
# - Using DCT instead of wavelet for 1st step; implement bior1.5 (is there a package for this?)
# - This implementation is rather slow, since it computes the entire 3D group spectrum at 
#   once.  A better implementation uses a rolling buffer to only compute the 3D groups in 
#   the current search window
# - Remove magic numbers / hard-coded parameters
# - Add Kaiser window to reduce border effects between patches

module BM3D
export bm3d_thr, bm3d_wie

using Hadamard
using NumericExtensions

# 1st step of BM3D denoising: hard thresholding
function bm3d_thr(img::Matrix{Float64}, sigma::Float64)

	# Hard code algorithm parameters for now...
	patchSize = [8;8] 
	stepSize = [3;3]  
	nBorder = [0;0]
	searchWin = [19;19]
	nMatch = 31
	thresh3D = 2.7

	# Block matching
	(Ilist,Jlist) = get_reference_pixels([size(img,1);size(img,2)],patchSize,stepSize,nBorder)
	matchTable = match_patches(img,Ilist,Jlist,patchSize,searchWin,nMatch)

	G3D = form_groups(img,matchTable,Ilist,Jlist,patchSize)

	# Filter 3D groups by hard thresholding 
	G3D[abs(G3D) .< sigma*thresh3D] = 0

	W = zeros(size(G3D))
	for j = 1:length(Jlist)
		for i = 1:length(Ilist)
			T = nnz(G3D[:,:,:,i,j])
			W[:,:,:,i,j] = T > 0 ? 1.0/T : 1.0
		end
	end
	G3D .*= W

	imgOut = invert_groups([size(img,1);size(img,2)],G3D,matchTable,Ilist,Jlist,patchSize) 

	Wout = zeros(Float64,size(img))
	groups_to_image!(Wout,W,matchTable,Ilist,Jlist,patchSize)

	return imgOut./Wout

end

# 2nd step of BM3D
# img: input noisy image
# imgBasic: denoised image from first step of BM3D (hard thresholding)
# sigma: known or assumed standard deviation of noise
function bm3d_wie(img::Matrix{Float64}, imgBasic::Matrix{Float64}, sigma::Float64)

	# Hard code algorithm parameters for now...
	patchSize = [8;8] 
	stepSize = [3;3]  
	nBorder = [0;0]
	searchWin = [11;11]
	nMatch = 15
	thresh3D = 2.7

	# block matching step
	(Ilist,Jlist) = get_reference_pixels([size(img,1);size(img,2)],patchSize,stepSize,nBorder)
	matchTable = match_patches(imgBasic,Ilist,Jlist,patchSize,searchWin,nMatch)

	# Compute 3D group spectrum
	G3D = form_groups(img,matchTable,Ilist,Jlist,patchSize)
	G3Dbasic = form_groups(imgBasic,matchTable,Ilist,Jlist,patchSize)

	# Wiener filtering of 3D groups, using basic estimate as target spectrum
	WC = G3Dbasic.^2./(G3Dbasic.^2 + sigma^2) # devec?
	G3D .*= WC

	# Weight groups 
	W = zeros(Float64,size(G3D))
	for j = 1:length(Jlist)
		for i = 1:length(Ilist)
			T = sumsq(WC[:,:,:,i,j])
			W[:,:,:,i,j] = T > 0 ? 1.0/T : 1.0
		end
	end

	G3D .*= W

	Wout = zeros(Float64,size(img))
	groups_to_image!(Wout,W,matchTable,Ilist,Jlist,patchSize)

	imgOut = invert_groups([size(img,1);size(img,2)],G3D,matchTable,Ilist,Jlist,patchSize) 

	return imgOut./Wout

end

# Forward BM3D groupings (full transform... inefficient!)
function form_groups(img::Matrix{Float64},
					 matchTable::Array{Float64,4},
					 Ilist::Vector{Int64},
					 Jlist::Vector{Int64},
					 patchSize::Vector{Int64})

	(t,Nmatch,N1,N2) = size(matchTable)

	G3D = zeros(Float64,Nmatch+1,patchSize[1],patchSize[2],N1,N2)

	# Form table of 3D groups
	image_to_groups!(img,G3D,matchTable,Ilist,Jlist,patchSize)

	# Apply 3D DCT on groups
	dct!(G3D,1:3)

	# Apply normalized Walsh-Hadamard transform on 3rd dimension of groups
	#G3D = fwht(G3D,1)*sqrt(float(Nmatch+1))

	return G3D

end

# Inverse BM3D
function invert_groups(imgSize::Vector{Int64},
					   G3D::Array{Float64,5},
					   matchTable::Array{Float64,4},
					   Ilist::Vector{Int64},
					   Jlist::Vector{Int64},
					   patchSize::Vector{Int64})

	(t,Nmatch,N1,N2) = size(matchTable)

	# Allocate image and weight table
	img = zeros(Float64,imgSize[1],imgSize[2])

	# Normalized inverse Walsh-Hadamard transform on 3rd dimension of groups
	#G3D = ifwht(G3D,1)/sqrt(float(Nmatch+1))

	# Apply inverse 3D DCT on groups
	idct!(G3D,1:3)

	groups_to_image!(img,G3D,matchTable,Ilist,Jlist,patchSize)

	return img

end

# Return filtered patches to their place in the image
function groups_to_image!(img::Matrix{Float64},
						  G3D::Array{Float64,5},
						  matchTable::Array{Float64,4},
						  Ilist::Vector{Int64},
						  Jlist::Vector{Int64},
						  patchSize::Vector{Int64})

	Nmatch = size(matchTable,2)

	for j1 = 1:length(Jlist)
		for i1 = 1:length(Ilist)

			for jj = 1:patchSize[2]
				for ii = 1:patchSize[1]
					img[Ilist[i1]+ii-1,Jlist[j1]+jj-1] += G3D[1,ii,jj,i1,j1]
				end
			end

			for k = 1:Nmatch

				i2 = i1 + matchTable[1,k,i1,j1]
				j2 = j1 + matchTable[2,k,i1,j1]

				for jj = 1:patchSize[2]
					for ii = 1:patchSize[1]
						img[Ilist[i2]+ii-1,Jlist[j2]+jj-1] += G3D[k+1,ii,jj,i1,j1]
					end
				end
			end

		end
	end
end

function image_to_groups!(img::Matrix{Float64},
	                      G3D::Array{Float64,5},
	                      matchTable::Array{Float64,4},
	                      Ilist::Vector{Int64},
	                      Jlist::Vector{Int64},
	                      patchSize::Vector{Int64})

	Nmatch = size(matchTable,2)

	for j1 = 1:length(Jlist)
		for i1 = 1:length(Ilist)
			for jj = 1:patchSize[2]
				for ii = 1:patchSize[1]
					G3D[1,ii,jj,i1,j1] = img[Ilist[i1]+ii-1,Jlist[j1]+jj-1]
				end
			end

			for k = 1:Nmatch

				i2 = i1 + matchTable[1,k,i1,j1]
				j2 = j1 + matchTable[2,k,i1,j1]

				for jj = 1:patchSize[2]
					for ii = 1:patchSize[1]
						G3D[k+1,ii,jj,i1,j1] = img[Ilist[i2]+ii-1,Jlist[j2]+jj-1]
					end
				end
			end
		end
	end
end

# Get locations of reference pixels, the upper-left corner of each patch.

# imgSize - size of input image, including border
# patchSize - size of patch in pixels
# stepSize - step in x and y between reference pixels
# nBorder - number of border pixels on each side in x and y

function get_reference_pixels(imgSize::Vector{Int64},
							  patchSize::Vector{Int64},
							  stepSize::Vector{Int64},
							  nBorder::Vector{Int64})

    ph = imgSize[1] - 2*nBorder[1] - patchSize[1] + 1
    pw = imgSize[2] - 2*nBorder[2] - patchSize[2] + 1

    I = [nBorder[1] + 1:stepSize[1]:ph]
    J = [nBorder[2] + 1:stepSize[2]:pw]

    # Make sure there is a patch touching the lower and right borders
    if(maximum(I) < nBorder[1] + ph)
        I = [I; nBorder[1] + ph]
    end
    if(maximum(J) < nBorder[2] + pw)
        J = [J; nBorder[2] + pw]
    end

    return (I,J)
    
end

# Full-search block matching algorithm for BM3D
function match_patches(img::Matrix{Float64},
					   Ilist::Vector{Int64},
					   Jlist::Vector{Int64},
					   patchSize::Vector{Int64},
					   searchWin::Vector{Int64},
					   nMatch::Int64)

	# Dimensions of patch table
	N1 = length(Ilist)
	N2 = length(Jlist)

	# Allocate 
	matchTable = zeros(Float64,(3,nMatch,N1,N2))
    matchMaxTable = ones(Float64,(2,N1,N2))
    matchTable[3,:,:,:] = realmax(Float64)
    matchMaxTable[2,:,:] = realmax(Float64)

    for j1 = 1:N2
    	for i1 = 1:N1
    	
    		for j2 = j1:minimum([N2;j1 + searchWin[2]])
    		# Lower bound on columns
    		LB = (j1 == j2) ? (i1+1) : (i1 - searchWin[1])
    		for i2 = maximum([1,LB]):minimum([i1+searchWin[1];N1])

    				d2 = 0.0
    				for jj = 1:patchSize[2]
    					for ii = 1:patchSize[1]
    						d2 += (img[Ilist[i1] + ii - 1,Jlist[j1] + jj - 1] - img[Ilist[i2] + ii - 1,Jlist[j2] + jj - 1])^2
    					end
    				end
    				d2 /= prod(patchSize)

    				 # Check current maximum for patch (i1,j1)
	                 if (d2 < matchMaxTable[2,i1,j1])
	                 	kmatch = matchMaxTable[1,i1,j1]
	                 	matchTable[1,kmatch,i1,j1] = i2-i1
	                 	matchTable[2,kmatch,i1,j1] = j2-j1
	                 	matchTable[3,kmatch,i1,j1] = d2

	                 	(tmp2,tmp1) = findmax(matchTable[3,:,i1,j1])
	                    matchMaxTable[1,i1,j1] = tmp1
	                    matchMaxTable[2,i1,j1] = tmp2
	                 end

	                 # Check current maximum for patch (i2,j2)
	                 if (d2 < matchMaxTable[2,i2,j2])
	                 	kmatch = matchMaxTable[1,i2,j2]
	                 	matchTable[1,kmatch,i2,j2] = i1-i2
	                 	matchTable[2,kmatch,i2,j2] = j1-j2
	                 	matchTable[3,kmatch,i2,j2] = d2

	                    (tmp2,tmp1) = findmax(matchTable[3,:,i2,j2])
	                    matchMaxTable[1,i2,j2] = tmp1
	                    matchMaxTable[2,i2,j2] = tmp2
	                 end

    			end
    		end

    	end
    end

	return matchTable

end

end #module

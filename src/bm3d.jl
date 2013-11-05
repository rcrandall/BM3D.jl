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
# - Remove magic numbers / hard-coded parameters
# - Add Kaiser window to reduce border effects between patches

module BM3D
export bm3d_thr, bm3d_wie

using Hadamard
using Devectorize
using NumericExtensions

# 1st step of BM3D denoising: hard thresholding
function bm3d_thr(img::Matrix{Float64}, sigma::Float64)

	# Hard code algorithm parameters for now...
	patchSize = 32
	stepSize = 32
	nBorder = 0 # unused for now
	searchWin = 3
	nMatch = 1
	thresh3D = 2.7

	imgOut = zeros(Float64,size(img))
	wOut = zeros(Float64,size(img))

	# Block matching
	(Ilist,Jlist) = get_reference_pixels([size(img,1);size(img,2)],patchSize,stepSize,nBorder)

	# Initialize match table 
	matchTable = zeros(Float64,3,nMatch,length(Ilist),searchWin+1)
	maxTable = ones(Float64,2,length(Ilist),searchWin+1)
    matchTable[3,:,:,:] = realmax(Float64)
    maxTable[2,:,:] = realmax(Float64)

	# Initialize patch table
	patchTable = zeros(Float64,patchSize,patchSize,length(Ilist),2*searchWin+1)

	# Initialize 3D groups
	G3D = zeros(Float64,nMatch+1,patchSize,patchSize,length(Ilist))
	W = zeros(Float64,size(G3D))

	for j = 1:length(Jlist)
		if (nMatch > 0)
			update_matches!(j,img,matchTable,maxTable,Ilist,Jlist,patchSize,searchWin,nMatch)
		end
		update_patches!(j,img,patchTable,Ilist,Jlist,patchSize,searchWin)
		
		form_groups!(G3D,patchTable,matchTable,Ilist,Jlist,patchSize,searchWin)

		# Filter 3D groups by hard thresholding 
		@devec G3D[abs(G3D) .< sigma*thresh3D] = 0

		# Compute weights
		for i = 1:length(Ilist)
			T = nnz(G3D[:,:,:,i])
			T2 = T > 0 ? 1.0/T : 1.0
			W[:,:,:,i] = 1
			G3D[:,:,:,i] *= 1
		end

		# Normalized inverse Walsh-Hadamard transform on 3rd dimension of groups
		G3D = ifwht(G3D,1)/sqrt(float(nMatch+1))

		# Apply inverse 2D DCT on each patch
		idct!(G3D,2:3)

		groups_to_image!(j,imgOut,G3D,matchTable,Ilist,Jlist,patchSize) 
		groups_to_image!(j,wOut,W,matchTable,Ilist,Jlist,patchSize)
	end

	return (imgOut./wOut,matchTable)

end

# Forward BM3D groupings for the current column
function form_groups!(G3D::Array{Float64,4},
	                  patchTable::Array{Float64,4},
					  matchTable::Array{Float64,4},
					  Ilist::Vector{Int64},
					  Jlist::Vector{Int64},
					  patchSize::Int64,
					  searchWin::Int64)

	(t,Nmatch,N1,N2) = size(matchTable)

	# Process current column
	for i1 = 1:length(Ilist)

		# Each patch self-matches
		for jj = 1:patchSize
			for ii = 1:patchSize
				G3D[1,ii,jj,i1] = patchTable[ii,jj,i1,searchWin+1]
			end
		end

		# Add other matched patches to the group
		for k = 1:Nmatch
			i2 = i1 + matchTable[1,k,i1,1]
			j2 = searchWin + 1 + matchTable[2,k,i1,1]
			for jj = 1:patchSize
				for ii = 1:patchSize
					G3D[k+1,ii,jj,i1] = patchTable[ii,jj,i2,j2]
				end
			end
		end
	end

	# Apply 2D DCT on each patch
	dct!(G3D,2:3)

	# Apply normalized Walsh-Hadamard transform along 3rd dimension of groups
	G3D = fwht(G3D,1)*sqrt(float(Nmatch+1))
end

# Return filtered patches to their place in the image
function groups_to_image!(jIn::Int64,
						  img::Matrix{Float64},
						  G3D::Array{Float64,},
						  matchTable::Array{Float64,4},
						  Ilist::Vector{Int64},
						  Jlist::Vector{Int64},
						  patchSize::Int64)

	Nmatch = size(matchTable,2)

	j1 = jIn
	
		# Process each patch in current column
		for i1 = 1:length(Ilist)

			# Each patch self-matches
			for jj = 1:patchSize
				for ii = 1:patchSize
					img[Ilist[i1]+ii-1,Jlist[j1]+jj-1] += G3D[1,ii,jj,i1]
				end
			end

			# 
			for k = 1:Nmatch
				i2 = i1 + matchTable[1,k,i1,1]
				j2 = j1 + matchTable[2,k,i1,1]

				for jj = 1:patchSize
					for ii = 1:patchSize
						img[Ilist[i2]+ii-1,Jlist[j2]+jj-1] += G3D[k+1,ii,jj,i1]
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
							  patchSize::Int64,
							  stepSize::Int64,
							  nBorder::Int64)

    ph = imgSize[1] - 2*nBorder - patchSize + 1
    pw = imgSize[2] - 2*nBorder - patchSize + 1

    I = [nBorder + 1:stepSize:ph]
    J = [nBorder + 1:stepSize:pw]

    # Make sure there is a patch touching the lower and right borders
    if(maximum(I) < nBorder + ph)
        I = [I; nBorder + ph]
    end
    if(maximum(J) < nBorder + pw)
        I = [J; nBorder + pw]
    end

    return (I,J)
    
end

# Full-search block matching algorithm for BM3D
function update_matches!(jIn::Int64,
						 img::Matrix{Float64},
					     matchTable::Array{Float64,4},
					     matchMaxTable::Array{Float64,3},
					     Ilist::Vector{Int64},
					     Jlist::Vector{Int64},
					     patchSize::Int64,
					     searchWin::Int64,
					     nMatch::Int64)

	# Dimensions of patch table
	N1 = length(Ilist)
	N2 = length(Jlist)

	if(jIn > 1)
		# Shift match table to reuse previous computations
		matchTable[:,:,:,1:searchWin] = matchTable[:,:,:,2:end]
		matchMaxTable[:,:,1:searchWin] = matchMaxTable[:,:,2:end]
		#matchTable[3,:,:,end] = realmax(Float64)
		#matchMaxTable[2,:,end] = realmax(Float64)
	end

	j1 = jIn
    	for i1 = 1:N1
    		for j2 = j1:minimum([N2;j1 + searchWin])
    		# Lower bound on rows: only search down in current row
    		LB = (j1 == j2) ? (i1+1) : (i1 - searchWin)
    		for i2 = maximum([1,LB]):minimum([i1+searchWin;N1])
    			if(i1 != i2 || j1 != j2)
    				d2 = 0.0
    				for jj = 1:patchSize
    					for ii = 1:patchSize
    						d2 += (img[Ilist[i1] + ii - 1,Jlist[j1] + jj - 1] 
    							 - img[Ilist[i2] + ii - 1,Jlist[j2] + jj - 1])^2
    					end
    				end
    				d2 /= prod(patchSize)

    				 # Check current maximum for patch (i1,j1)
	                 if (d2 < matchMaxTable[2,i1,1])
	                 	kmatch = matchMaxTable[1,i1,1]
	                 	matchTable[1,kmatch,i1,1] = i2-i1
	                 	matchTable[2,kmatch,i1,1] = j2-j1
	                 	matchTable[3,kmatch,i1,1] = d2

	                 	(tmp2,tmp1) = findmax(matchTable[3,:,i1,1])
	                    matchMaxTable[1,i1,1] = tmp1
	                    matchMaxTable[2,i1,1] = tmp2
	                 end

	                 # Check current maximum for patch (i2,j2)
	                 if (d2 < matchMaxTable[2,i2,j2-jIn+1])
	                 	kmatch = matchMaxTable[1,i2,j2-jIn+1]
	                 	matchTable[1,kmatch,i2,j2-jIn+1] = i1-i2
	                 	matchTable[2,kmatch,i2,j2-jIn+1] = j1-j2
	                 	matchTable[3,kmatch,i2,j2-jIn+1] = d2

	                    (tmp2,tmp1) = findmax(matchTable[3,:,i2,j2-jIn+1])
	                    matchMaxTable[1,i2,j2-jIn+1] = tmp1
	                    matchMaxTable[2,i2,j2-jIn+1] = tmp2
	                 end
					end
    			end
    		end

    	end
end

function update_patches!(jIn::Int64,
						 img::Matrix{Float64},
						 patchTable::Array{Float64,4},
						 Ilist::Vector{Int64},
						 Jlist::Vector{Int64},
						 patchSize::Int64,
						 searchWin::Int64)

	# If processing first column, compute all patches to the right within the search window
	if(jIn == 1)
		for j = 1:min(searchWin+1,length(Jlist))
			for i = 1:length(Ilist)
				patchTable[:,:,i,searchWin + j] = img[Ilist[i]:(Ilist[i]+patchSize-1),Jlist[j]:(Jlist[j]+patchSize-1)]
			end
		end

	# If past first column, reuse patches we've already looked at and get the patches in the rightmost column
	else
		# Shift table...
		patchTable[:,:,:,1:(2*searchWin)] = patchTable[:,:,:,2:(2*searchWin+1)]

		if(jIn + searchWin <= length(Jlist))
			for i = 1:length(Ilist)
				patchTable[:,:,i,end] = img[Ilist[i]:(Ilist[i]+patchSize-1),Jlist[jIn+searchWin]:(Jlist[jIn+searchWin]+patchSize-1)]
			end
		end
	end

end

end #module
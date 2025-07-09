#include <algorithm>
#include <array>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>

#include "mex.h"

using Coordinates = std::array<double,3>;

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
/*
 *  PARSE_GMSH_OUTPUT extracts the FE mesh definition from an GMSH's output file
 * 
 *  [ nodes, elements, elementTypes ] = parse_GMSH_output( filePath )
 * 
 *  Features:   * Extracts also combined meshes but warns if different element types are encountered 
 *              * Extracts only elements defined for the larges identified dimensionality (see maxDimension)
 * 
 *  Author:  Martin Doskar (MartinDoskar@gmail.com)     
 *  Version: 0.1 (2019-04-08)
 */
{
    if (nrhs != 1)
        mexErrMsgTxt("Path to the input file is required.");

    // Open file
    auto filePath = std::string{ mxArrayToString(prhs[0]) };
    auto iFile = std::ifstream{ filePath };
	if (iFile.fail())
		mexErrMsgTxt("Unable to open the input file.");

    // Parse header	
    auto sectionLabel = std::string{};
	{
		auto formatVersion = 0.0;
		auto isBinary = false;
		auto size_tSize = 0;

		iFile >> sectionLabel;
		iFile >> formatVersion >> isBinary >> size_tSize;
		iFile >> sectionLabel;

		if (formatVersion < 4.0 || isBinary)
			mexErrMsgTxt("Only ASCII variant of the MSH format version 4.0 and newer is supported.");
	}

    // Parse nodal data
	while (sectionLabel.compare("$Nodes") && !iFile.eof())
		iFile >> sectionLabel;

	if (iFile.eof())
		mexErrMsgTxt("MSH file must contain '$Nodes' section.");

    auto parsedNodes = std::vector<Coordinates>{};
    auto maxDimension = 0;				// Keep track of highest dimension encountered in the analysis
	{
		auto nEntityBlocks = 0;
		auto nNodes = 0;
		auto minNodalInd = 0;
		auto maxNodalInd = 0;
		iFile >> nEntityBlocks >> nNodes >> minNodalInd >> maxNodalInd;

		if (nNodes != maxNodalInd)
			mexErrMsgTxt("Nodes are not number consecutively.");

		parsedNodes.resize(maxNodalInd, Coordinates{0.0,0.0,0.0});

		for (auto iB = 0; iB < nEntityBlocks; ++iB) {
			auto entityDim = 0;
			auto entityTag = 0;
			auto isParametric = false;
			auto nEntityNodes = static_cast<size_t>(0);

			iFile >> entityDim >> entityTag >> isParametric >> nEntityNodes;

			maxDimension = std::max(maxDimension, entityDim);	

			if (isParametric)
				mexErrMsgTxt("Parametric coordinates are not supported.");

			auto nodalIds = std::vector<size_t>(nEntityNodes, 0);
			for (auto iN = static_cast<size_t>(0); iN < nEntityNodes; ++iN) {
				auto ind = static_cast<size_t>(0);
				iFile >> ind;
				nodalIds.at(iN) = (ind - 1);
			}
			for (auto iN = static_cast<size_t>(0); iN < nEntityNodes; ++iN) {
				auto coords = Coordinates{};
				iFile >> coords.at(0) >> coords.at(1) >> coords.at(2);
				parsedNodes.at(nodalIds.at(iN)) = std::move(coords);
			}
		}
	}

    // Parse element data
	while (sectionLabel.compare("$Elements") && !iFile.eof())
		iFile >> sectionLabel;

	if (iFile.eof())
		mexErrMsgTxt("MSH file must contain '$Elements' section.");

	auto parsedElements = std::vector<std::vector<int>>{};
	auto parsedElementTypes = std::vector<int>{};
	{
		auto nEntityBlocks = 0;
		auto nElements = 0;
		auto minElementInd = 0;
		auto maxElementInd = 0;

		iFile >> nEntityBlocks >> nElements >> minElementInd >> maxElementInd;

		for (auto iB = 0; iB < nEntityBlocks; ++iB) {
			auto entityDim = 0;
			auto entityTag = 0;
			auto elementType = 0;
			auto nEntityElements = static_cast<size_t>(0);

			iFile >> entityDim >> entityTag >> elementType >> nEntityElements;
			iFile.ignore(std::numeric_limits<std::streamsize>::max(), '\n');	// End the previous line

			if (entityDim == maxDimension) {
				for (auto iE = static_cast<size_t>(0); iE < nEntityElements; ++iE) {
					auto lineString = std::string{};

					std::getline(iFile, lineString);
					auto lineStream = std::stringstream{ lineString };

					auto elementId = 0;
					lineStream >> elementId;

					auto elementNodes = std::vector<int>{};
					auto nodeInd = 0;
					while (lineStream >> nodeInd)
						elementNodes.push_back(nodeInd );

                    parsedElements.push_back(std::move(elementNodes));
                    parsedElementTypes.push_back(elementType);
				}
			}
			else {
				for (auto iE = static_cast<size_t>(0); iE < nEntityElements; ++iE)
					iFile.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
			}

		}
	}

    // Reshuffle data into the output format
    const auto nNodes = parsedNodes.size();
    const auto nElems = parsedElements.size();
    const auto nMaxNodesPerElem = std::max_element(std::begin(parsedElements), std::end(parsedElements),
        [](const auto& a, const auto& b){ return a.size() < b.size(); })->size();

    if (std::any_of(std::begin(parsedElementTypes), std::end(parsedElementTypes), [&parsedElementTypes](const auto& v){ return v != parsedElementTypes.at(0); }))
        mexWarnMsgTxt("Mesh with mixed elements identified");

    plhs[0] = mxCreateDoubleMatrix(static_cast<mwSize>(3), static_cast<mwSize>(nNodes), mxREAL);
	double* ptrNodes = mxGetPr(plhs[0]);
    for (auto iN = 0; iN < nNodes; ++iN) {
        for (auto iC = 0; iC < 3; ++iC) {
            ptrNodes[iN*3 + iC] = parsedNodes.at(iN).at(iC);
        }
    }

    plhs[1] = mxCreateDoubleMatrix(static_cast<mwSize>(nMaxNodesPerElem), static_cast<mwSize>(nElems), mxREAL);
	double* ptrElems = mxGetPr(plhs[1]);
    plhs[2] = mxCreateDoubleMatrix(static_cast<mwSize>(1), static_cast<mwSize>(nElems), mxREAL);
	double* ptrTypes = mxGetPr(plhs[2]);
    for (auto iE = 0; iE < nElems; ++iE) {
        ptrTypes[iE] = parsedElementTypes.at(iE);
        for (auto iN = 0; iN < parsedElements.at(iE).size(); ++iN) {
            ptrElems[iE*nMaxNodesPerElem + iN] = parsedElements.at(iE).at(iN);
        }
    }

}
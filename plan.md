# SUSRUTA: System for Unified Graph‐based Heuristic Recommendation Using Treatment Analytics
## A Graph-Based Glioma Treatment Outcome Prediction System

## 1. Refined Problem Statement

### 1.1 Core Challenge
Oncologists treating glioma patients face critical decision points after initial treatment, particularly when determining whether to continue current therapy, adjust dosages, switch modalities, or consider experimental options. Current decision-making relies heavily on physician experience and general guidelines that fail to account for the complex interplay between patient-specific characteristics, tumor biology, genetic profiles, and treatment history.

### 1.2 Precise Objective
Develop a memory-efficient, graph-based clinical decision support system that:

1. Predicts patient-specific responses to candidate treatment regimens
2. Provides actionable, personalized therapeutic recommendations 
3. Quantifies prediction uncertainty and explains reasoning
4. Accounts for the complex interrelationships between tumor characteristics, treatment modalities, and temporal progression patterns

### 1.3 Target Clinical Scenarios
- **Primary Use Case**: Post-initial treatment decision points (particularly at 3-month and 6-month follow-up intervals)
- **Secondary Use Case**: Treatment planning following tumor recurrence
- **Target Users**: Neuro-oncology tumor boards, radiation oncologists, neuro-oncologists

### 1.4 Technical Constraints
- Must function within 8GB RAM limit of M1 MacBook Air
- Must process a single patient case in under 5 minutes
- Must handle missing data scenarios common in clinical settings

## 2. Data Engineering Framework

### 2.1 Dataset Partitioning Strategy
- **Training Set**: 140 patients (70%)
- **Validation Set**: 30 patients (15%)
- **Test Set**: 33 patients (15%)
- **Stratification Factors**: Distribution of treatment outcomes, tumor types, genetic profiles

### 2.2 Memory-Efficient MRI Processing Pipeline

#### 2.2.1 MRI Preprocessing Protocol
1. **Efficient NIfTI Loading**:
   - Implement lazy loading approach that reads slices on demand
   - Use SimpleITK for memory-optimized I/O operations
   - Store 16-bit integer representations where possible

2. **Progressive Downsampling Strategy**:
   - Initial global feature extraction at 25% resolution (64×64×64)
   - Extract tumor region and process at 50% resolution (128×128×128)
   - Process critical ROIs at full resolution using sliding window approach
   
3. **Feature Extraction with Memory Constraints**:
   - Extract 3D patches around tumor regions rather than whole volumes
   - Use chunked processing to stay within memory limits
   - Implement PyTorch memory optimization techniques:
     - Use in-place operations where possible
     - Free unused tensors explicitly
     - Utilize mixed precision (float16) for feature extraction
     - Implement gradient checkpointing

#### 2.2.2 Radiomics Feature Extraction 
*Instead of processing raw volumes directly, extract compact radiomic features to reduce computational overhead*

1. **First-order Statistics**:
   - Mean, median, variance, skewness, kurtosis, entropy (per volume)
   
2. **Shape/Morphological Features** (from tumorMask):
   - Volume, surface area, compactness, sphericity
   - Maximum 3D diameter, major/minor axis lengths
   
3. **Texture Features**:
   - GLCM-based features (contrast, correlation, homogeneity)
   - GLRLM-based features (run length, run percentage)
   - Limit to 2D computation on representative slices to save memory
   
4. **Spatial Features**:
   - Distance to ventricles, critical structures
   - Invasion metrics (contact with white matter tracts)
   - Optimize calculation using sparse representations

### 2.3 Clinical Data Processing

1. **Categorical Variable Handling**:
   - One-hot encoding for treatment modalities
   - Ordinal encoding for tumor grades
   - Entity embedding for high-cardinality features
   
2. **Numerical Feature Preprocessing**:
   - Robust scaling (mitigate outlier effects)
   - Missing value imputation via k-nearest neighbors
   
3. **Temporal Data Structuring**:
   - Interval-based encoding of treatment sequences
   - Event timing relative to diagnosis
   - Treatment duration and cumulative dose calculation
   
4. **Clinical Feature Engineering**:
   - Treatment response indicators at each timepoint
   - Composite biomarker indexes
   - Derived patient risk scores

### 2.4 Multi-modal Data Integration Strategy

1. **Feature-level Fusion**:
   - Aligned temporal representation across modalities
   - Normalization within modalities before combination
   
2. **Graph Structure Integration**:
   - Connect clinical nodes to relevant imaging features
   - Weight edges based on statistical correlation
   - Create hierarchical feature groups to reduce dimensionality

## 3. Knowledge Graph Construction

### 3.1 Graph Ontology Design

#### 3.1.1 Node Types and Attributes
1. **Patient Nodes** (n=203):
   - Demographics (age, sex)
   - Baseline performance status
   - Prior treatment history
   
2. **Tumor Nodes** (multiple per patient):
   - Location (anatomical region)
   - Grade and histological type
   - Molecular subtype
   - Growth pattern
   
3. **Treatment Nodes**:
   - **Surgery Nodes**: Extent of resection, approach
   - **Radiation Nodes**: Protocol, dose, fractionation, technique
   - **Chemotherapy Nodes**: Agent, dosage, duration, schedule
   - **Combination Therapy Nodes**: Concurrent treatments
   
4. **Outcome Nodes**:
   - Radiographic response
   - Progression-free interval
   - Overall survival
   - Treatment toxicity
   
5. **Timepoint Nodes**:
   - MRI acquisition date
   - Days from initial diagnosis
   - Days from treatment initiation
   
6. **Imaging Feature Nodes**:
   - Radiomic features
   - Segmentation-derived metrics
   - Contrast enhancement patterns

#### 3.1.2 Edge Types and Relationships
1. **Patient-Tumor Edges**:
   - Attributes: Date of diagnosis, presenting symptoms
   
2. **Tumor-Treatment Edges**:
   - Attributes: Treatment intent, treatment order
   
3. **Treatment-Outcome Edges**:
   - Attributes: Response evaluation criteria, toxicity grade
   
4. **Temporal Edges**:
   - Connect sequential timepoints
   - Connect treatments to follow-up scans
   
5. **Feature Similarity Edges**:
   - Connect patients with similar tumor characteristics
   - Connect treatments with similar response patterns
   
6. **Causal Inference Edges**:
   - Estimated treatment effect
   - Confidence in causal relationship

### 3.2 Efficient Graph Construction Algorithm

1. **Initialization Phase**:
   ```python
   # Memory-efficient graph creation
   G = nx.MultiDiGraph()  # Use directed multigraph to support multiple edge types
   
   # Create nodes in batches to control memory usage
   for batch in data_loader(batch_size=20):  # Process 20 patients at a time
       # Add patient nodes
       patient_nodes = [(f"patient_{id}", patient_attrs) for id, patient_attrs in batch]
       G.add_nodes_from(patient_nodes, type="patient")
       
       # Process related entities (treatments, outcomes, etc.)
       # ...
   ```

2. **Graph Sparsification Techniques**:
   - Set threshold for edge creation based on statistical significance
   - Use approximate nearest neighbors for similarity connections
   - Prune redundant relationships
   - Implement edge sampling for large clusters

3. **Hierarchical Graph Structure**:
   - Create meta-nodes for treatment protocols
   - Group similar patients into cohorts
   - Develop multi-level graph to reduce complexity

### 3.3 Dynamic Graph Updating

1. **Incremental Updates**:
   - Add new patients without full graph reconstruction
   - Update outcome nodes as follow-up data becomes available
   
2. **Memory-Optimized Storage**:
   - Sparse adjacency matrix representation
   - Compressed attribute storage
   - On-disk graph database with in-memory processing

## 4. Model Architecture Design

### 4.1 Memory-Efficient GNN Formulation

#### 4.1.1 Base Architecture
```python
class GliomaGNN(torch.nn.Module):
    def __init__(self, node_features, edge_features, hidden_channels=32):
        super().__init__()
        
        # Node type-specific encoders (reduces dimensionality early)
        self.patient_encoder = nn.Linear(node_features["patient"], hidden_channels)
        self.treatment_encoder = nn.Linear(node_features["treatment"], hidden_channels)
        self.tumor_encoder = nn.Linear(node_features["tumor"], hidden_channels)
        
        # Heterogeneous graph convolution - mini-batch compatible
        self.conv1 = HeteroDynamicEdgeConv(hidden_channels, hidden_channels, aggr='mean')
        self.conv2 = HeteroDynamicEdgeConv(hidden_channels, hidden_channels, aggr='mean')
        
        # Treatment-outcome prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels*2, hidden_channels),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_channels, 1)
        )
        
    def forward(self, x_dict, edge_index_dict, batch_size=4):
        # Encode node features by type
        x_dict = {
            'patient': self.patient_encoder(x_dict['patient']),
            'treatment': self.treatment_encoder(x_dict['treatment']),
            'tumor': self.tumor_encoder(x_dict['tumor']),
            # ...other node types
        }
        
        # Process in mini-batches to save memory
        for i in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[i:i+batch_size]
            batch_x = x[batch_ids]
            # Mini-batch subgraph extraction
            # ...
            
        # Apply graph convolutions
        x_dict = self.conv1(x_dict, edge_index_dict)
        x_dict = {key: F.elu(x) for key, x in x_dict.items()}
        x_dict = self.conv2(x_dict, edge_index_dict)
        
        # Predict treatment outcomes
        # ...
        
        return treatment_outcomes, uncertainty_estimates
```

#### 4.1.2 Memory Optimization Techniques

1. **Mini-batch Processing**:
   - Implement neighborhood sampling strategy
   - Use subgraph extraction for batch processing
   - Dynamic batch sizing based on available memory

2. **Model Compression**:
   - Reduced precision (16-bit) for certain operations
   - Knowledge distillation from larger models
   - Parameter sharing across similar node types

3. **Efficient Message Passing**:
   - Sparse tensor operations
   - Early fusion of features to reduce intermediate results
   - Custom CUDA kernels for M1 optimization

### 4.2 Multi-Stage Training Strategy

1. **Pre-training Phase**:
   - Train node embedding models on subgraphs
   - Use contrastive learning to build robust representations
   - Self-supervised link prediction to capture relationship patterns

2. **Main Training Phase**:
   - Freeze pre-trained components
   - Train treatment response prediction with clinical supervision
   - Implement gradient accumulation for effective larger batches

3. **Fine-tuning Phase**:
   - Unfreeze critical layers
   - Adjust to specific treatment types
   - Optimize for calibrated uncertainty

### 4.3 Model Ensemble Strategy

1. **Specialized Sub-models**:
   - Treatment-specific prediction models
   - Temporal progression models
   - Survival prediction models

2. **Ensemble Integration**:
   - Weighted prediction averaging
   - Bayesian model combination
   - Stacking with meta-learner

### 4.4 Adaptive Inference Technique

1. **Early-exit Architecture**:
   - Multiple output heads at different layers
   - Confidence-based computation termination
   - Adaptive computation based on case complexity

2. **Resource-aware Inference**:
   - Dynamic resolution based on available memory
   - Prioritized feature computation
   - Caching of intermediate representations

## 5. Counterfactual Treatment Analysis Module

### 5.1 Treatment Effect Modeling

1. **Causal Inference Framework**:
   - Implement graph-based potential outcomes framework
   - Propensity score matching on graph structure
   - Doubly-robust estimation techniques

2. **Treatment Effect Estimation**:
   - Individual treatment effect (ITE) calculation
   - Confidence interval estimation
   - Heterogeneous treatment effect detection

### 5.2 Treatment Option Simulation

1. **Hypothetical Treatment Graphs**:
   - Generate alternative treatment plan graphs
   - Simulate outcomes under different protocols
   - Compare likely progression trajectories

2. **Treatment Ranking System**:
   - Multi-objective ranking (survival, quality of life, toxicity)
   - Personalized weighting based on patient preferences
   - Uncertainty-aware ordering

### 5.3 Memory-Efficient Implementation

1. **Optimization Techniques**:
   - Reuse base patient embeddings across simulations
   - Incremental graph updates rather than full reconstructions
   - Sparse difference tracking between treatment scenarios

## 6. Explainability and Visualization Framework

### 6.1 Treatment Decision Explanations

1. **Feature Attribution Methods**:
   - Graph-specific SHAP values
   - Attention weight visualization
   - Counterfactual explanations

2. **Comparative Treatment Analysis**:
   - Side-by-side visualization of treatment options
   - Risk-benefit tradeoff charts
   - Historical patient comparisons

### 6.2 Clinical Interpretation Tools

1. **Interactive Visualization Components**:
   - Treatment response probability curves
   - Survival estimation plots
   - Uncertainty visualization

2. **Clinician-oriented Dashboards**:
   - Summary statistics for tumor board presentation
   - Key actionable insights
   - Confidence indicators for predictions

### 6.3 Implementation Using Lightweight Libraries

1. **Visualization Stack**:
   - Matplotlib for static plots (memory-efficient)
   - Lightweight interactive components with Streamlit
   - Server-side rendering for complex visualizations

## 7. Evaluation Framework

### 7.1 Performance Metrics

1. **Predictive Accuracy Metrics**:
   - Area under ROC curve for response prediction
   - Concordance index for survival outcomes
   - Brier score for calibration assessment

2. **Treatment Recommendation Quality**:
   - Retrospective policy evaluation
   - Expert agreement rate
   - Decision impact assessment

3. **Computational Efficiency Metrics**:
   - Peak memory usage tracking
   - Inference time benchmarking
   - Model size and compression ratio

### 7.2 Clinical Relevance Assessment

1. **Oncologist Feedback Protocol**:
   - Blinded comparison with expert recommendations
   - Qualitative assessment of explanation quality
   - Usefulness rating for clinical decision-making

2. **Case Study Evaluation**:
   - Detailed analysis of representative cases
   - Assessment of counterfactual reasoning
   - Comparison with actual treatment outcomes

### 7.3 Benchmarking Framework

1. **Baseline Models**:
   - Traditional machine learning methods (RF, XGBoost)
   - Non-graph deep learning approaches
   - Current clinical guidelines

2. **Comparative Analysis**:
   - Feature ablation studies
   - Graph structure importance
   - Multimodal fusion benefits

## 8. Implementation Roadmap

### 8.1 Development Phases

#### Phase 1: Core Data Processing Infrastructure (Weeks 1-3)
- Setup efficient MRI processing pipeline
- Implement clinical data preprocessing
- Develop feature extraction modules
- *Milestone*: Process full dataset within memory constraints

#### Phase 2: Graph Construction & Base Model (Weeks 4-6)
- Build knowledge graph structure
- Implement memory-efficient GNN architecture
- Develop training infrastructure
- *Milestone*: Train initial prediction model

#### Phase 3: Treatment Effect Modeling (Weeks 7-9)
- Implement counterfactual reasoning module
- Develop treatment simulation framework
- Create treatment ranking system
- *Milestone*: Generate personalized treatment recommendations

#### Phase 4: Explainability & Refinement (Weeks 10-12)
- Build explanation generation system
- Develop visualization components
- Refine model based on evaluations
- *Milestone*: Complete end-to-end system with explanations

### 8.2 Memory Optimization Schedule

1. **Profiling Phase** (Early Development):
   - Identify memory bottlenecks
   - Benchmark critical operations
   - Establish memory budgets per component

2. **Optimization Implementation** (Mid Development):
   - Apply identified optimizations
   - Refactor high-memory operations
   - Implement caching strategies

3. **System Integration** (Late Development):
   - Balance resource allocation across components
   - Implement dynamic resource management
   - Optimize end-to-end pipeline

### 8.3 Testing Strategy

1. **Unit Testing Framework**:
   - Test individual components in isolation
   - Verify memory usage during operations
   - Ensure correctness of critical functions

2. **Integration Testing**:
   - Validate end-to-end pipeline
   - Verify data flow between components
   - Test on representative subsets

3. **Clinical Validation**:
   - Test on held-out patient cases
   - Compare with expert decisions
   - Validate explanation quality

## 9. Technical Challenges and Solutions

### 9.1 Memory Constraint Mitigations

| Challenge | Mitigation Strategy |
|-----------|---------------------|
| Large MRI volumes | Patch-based processing, progressive resolution approach |
| Full graph in memory | Mini-batch training, neighborhood sampling, subgraph extraction |
| Model parameter size | Knowledge distillation, parameter sharing, quantization |
| Intermediate activations | Gradient checkpointing, early feature fusion, in-place operations |
| Visualization memory | Server-side rendering, progressive loading, simplified plots |

### 9.2 Performance Optimization Techniques

1. **Computation Reduction**:
   - Early stopping for convergent subgraphs
   - Feature selection to reduce dimensionality
   - Sparse operations for graph processing

2. **Memory-Performance Tradeoffs**:
   - Accuracy vs. memory usage analysis
   - Runtime vs. memory consumption balance
   - Quality vs. resource allocation decisions

### 9.3 Algorithmic Innovations

1. **Memory-Efficient GNN Variants**:
   - Develop sparse attention mechanisms
   - Implement hierarchical graph pooling
   - Design progressive message passing schemes

2. **Custom Data Structures**:
   - Compressed sparse graph representations
   - Memory-mapped tensor storage
   - Adaptive precision containers

## 10. Clinical Integration and Impact

### 10.1 Deployment Strategy

1. **Standalone Application**:
   - Electron-based desktop application
   - Optimized for M1 architecture
   - Local processing of sensitive data

2. **Integration Points**:
   - DICOM compatibility for image import
   - Structured data import from EMR systems
   - Report generation for clinical documentation

### 10.2 Clinical Workflow Integration

1. **Use Case Scenarios**:
   - Pre-tumor board preparation
   - Patient consultation decision support
   - Multidisciplinary review augmentation

2. **Decision Support Format**:
   - Actionable treatment recommendations
   - Evidence-based justifications
   - Uncertainty quantification

### 10.3 Impact Assessment Framework

1. **Clinical Utility Metrics**:
   - Time saved in treatment planning
   - Confidence in decision-making
   - Consistency of recommendations

2. **Outcome Improvement Potential**:
   - Projected survival benefit
   - Reduction in adverse treatment effects
   - Quality of life enhancement

## 11. Necessary Tools and Resources

### 11.1 Software Stack

1. **Core Libraries**:
   - PyTorch (base ML framework)
   - PyTorch Geometric (GNN implementation)
   - NetworkX (graph analysis and construction)
   - SimpleITK (medical image processing)
   - PyRadiomics (feature extraction)
   - SHAP (explainability)
   - Streamlit (lightweight UI)

2. **Development Environment**:
   - Conda for environment management
   - Git for version control
   - pytest for testing
   - memory-profiler for optimization

### 11.2 Hardware Considerations

1. **Development Optimizations for M1**:
   - Use MPS acceleration for PyTorch
   - Optimize memory allocation for M1 architecture
   - Leverage efficient core scheduling

2. **External Computing Strategy**:
   - Preprocessing pipeline on external resources
   - Model training on cloud platforms
   - Local inference optimization

### 11.3 Data Management

1. **Efficient Storage Strategy**:
   - Compressed NIfTI format for images
   - Sparse matrix storage for features
   - Incremental graph updates

2. **Data Versioning**:
   - Track preprocessing parameters
   - Version feature extraction configurations
   - Document graph construction choices

## 12. Appendix: Implementation Details

### 12.1 Graph Construction Code Snippet

```python
def construct_treatment_graph(clinical_data, imaging_features, patient_outcomes, memory_limit_mb=6000):
    """
    Constructs a heterogeneous treatment graph with memory constraints.
    
    Args:
        clinical_data: DataFrame with patient clinical information
        imaging_features: Dict mapping patient_ids to imaging features
        patient_outcomes: DataFrame with outcome information
        memory_limit_mb: Memory limit in MB (default: 6000)
        
    Returns:
        G: NetworkX MultiDiGraph with treatment graph
    """
    # Initialize empty graph
    G = nx.MultiDiGraph()
    
    # Track memory usage
    process = psutil.Process()
    
    # Process in batches to control memory
    patient_batches = np.array_split(clinical_data.patient_id.unique(), 10)
    
    for batch in patient_batches:
        # Add patient nodes
        batch_clinical = clinical_data[clinical_data.patient_id.isin(batch)]
        
        for _, patient in batch_clinical.iterrows():
            # Add patient node
            G.add_node(f"patient_{patient.patient_id}", 
                      type="patient",
                      age=patient.age,
                      sex=patient.sex,
                      performance_status=patient.karnofsky)
            
            # Add tumor nodes
            tumor_id = f"tumor_{patient.patient_id}"
            G.add_node(tumor_id,
                      type="tumor",
                      grade=patient.grade,
                      location=patient.location,
                      molecular_subtype=patient.idh_status)
            
            # Connect patient to tumor
            G.add_edge(f"patient_{patient.patient_id}", tumor_id, 
                      relation="has_tumor")
            
            # Add imaging feature connections - use sparse representation
            if patient.patient_id in imaging_features:
                # Add only top K most important features to save memory
                top_features = select_top_features(imaging_features[patient.patient_id], k=20)
                
                for feature_name, value in top_features.items():
                    feature_id = f"feature_{feature_name}"
                    # Add feature node if not exists
                    if feature_id not in G:
                        G.add_node(feature_id, type="feature", name=feature_name)
                    
                    # Connect tumor to feature
                    G.add_edge(tumor_id, feature_id, weight=value)
            
            # Add treatment nodes and connections
            treatments = get_patient_treatments(patient.patient_id)
            for treatment in treatments:
                treatment_id = f"treatment_{treatment.id}"
                G.add_node(treatment_id,
                          type="treatment",
                          category=treatment.category,
                          protocol=treatment.protocol,
                          dose=treatment.dose)
                
                # Connect tumor to treatment
                G.add_edge(tumor_id, treatment_id, relation="treated_with")
                
                # Add outcome node and connection
                if has_outcome(patient.patient_id, treatment.id):
                    outcome = get_outcome(patient.patient_id, treatment.id)
                    outcome_id = f"outcome_{patient.patient_id}_{treatment.id}"
                    
                    G.add_node(outcome_id,
                              type="outcome",
                              response=outcome.response,
                              survival_days=outcome.survival_days,
                              progression=outcome.progression)
                    
                    # Connect treatment to outcome
                    G.add_edge(treatment_id, outcome_id, relation="resulted_in")
        
        # Check memory usage and optimize if needed
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
        if memory_usage_mb > memory_limit_mb * 0.8:
            # Apply graph compression techniques
            G = compress_graph(G)
    
    # Build inter-patient similarity edges
    G = add_similarity_edges(G, threshold=0.7, max_edges=5)
    
    return G
```

### 12.2 Memory-Efficient Feature Extraction

```python
def extract_imaging_features(nifti_path, mask_path, memory_limit_mb=2000):
    """
    Memory-efficient extraction of imaging features from MRI data.
    
    Args:
        nifti_path: Path to MRI NIfTI file
        mask_path: Path to tumor mask NIfTI file
        memory_limit_mb: Memory limit in MB
        
    Returns:
        features: Dictionary of extracted features
    """
    # Calculate maximum chunk size based on memory constraints
    img = sitk.ReadImage(nifti_path)
    mask = sitk.ReadImage(mask_path)
    
    # Convert to numpy for processing, but only when needed
    spacing = img.GetSpacing()
    size = img.GetSize()
    
    # Get bounding box of tumor to avoid processing whole image
    bbox_min, bbox_max = compute_bounding_box(mask)
    
    # Add margin to bounding box
    margin = 10  # 10 voxels margin
    bbox_min = [max(0, x - margin) for x in bbox_min]
    bbox_max = [min(s-1, x + margin) for x, s in zip(bbox_max, size)]
    
    # Extract region of interest only
    roi_size = [bbox_max[i] - bbox_min[i] + 1 for i in range(3)]
    
    # Determine if we need to process in chunks
    voxel_size_bytes = 4  # 32-bit float
    roi_memory_mb = np.prod(roi_size) * voxel_size_bytes / (1024 * 1024)
    
    features = {}
    
    if roi_memory_mb < memory_limit_mb * 0.5:
        # Can process ROI at once
        roi_img = sitk.RegionOfInterest(img, roi_size, bbox_min)
        roi_mask = sitk.RegionOfInterest(mask, roi_size, bbox_min)
        
        # Extract features
        features = compute_features(roi_img, roi_mask)
    else:
        # Need to process in chunks
        chunk_features = []
        
        # Divide into Z-direction chunks
        chunk_count = max(1, int(np.ceil(roi_memory_mb / (memory_limit_mb * 0.5))))
        chunk_size = int(np.ceil(roi_size[2] / chunk_count))
        
        for z_start in range(bbox_min[2], bbox_max[2] + 1, chunk_size):
            z_end = min(z_start + chunk_size - 1, bbox_max[2])
            chunk_bbox_min = [bbox_min[0], bbox_min[1], z_start]
            chunk_roi_size = [roi_size[0], roi_size[1], z_end - z_start + 1]
            
            # Extract chunk
            chunk_img = sitk.RegionOfInterest(img, chunk_roi_size, chunk_bbox_min)
            chunk_mask = sitk.RegionOfInterest(mask, chunk_roi_size, chunk_bbox_min)
            
            # Compute features for chunk
            chunk_features.append(compute_features(chunk_img, chunk_mask))
            
            # Explicitly delete to free memory
            del chunk_img
            del chunk_mask
            gc.collect()
        
        # Combine chunk features
        features = combine_chunk_features(chunk_features)
    
    return features
```

### 12.3 Memory-Efficient GNN Training

```python
def train_gnn_model(graph, features, labels, device, memory_limit_mb=6000):
    """
    Memory-efficient training of GNN for treatment outcome prediction.
    
    Args:
        graph: NetworkX graph
        features: Node features
        labels: Treatment outcome labels
        device: PyTorch device
        memory_limit_mb: Memory limit in MB
        
    Returns:
        model: Trained GNN model
    """
    # Convert to PyTorch Geometric data
    data = from_networkx(graph)
    
    # Add node features
    for node_type, node_features in features.items():
        node_idx = torch.tensor([i for i, node in enumerate(graph.nodes()) 
                               if graph.nodes[node]['type'] == node_type])
        if len(node_idx) > 0:
            data[f'{node_type}_x'] = torch.FloatTensor(node_features)
    
    # Initialize model
    model = GliomaGNN(node_features=get_feature_dims(features),
                    edge_features=get_edge_feature_dims(graph),
                    hidden_channels=32)  # Small hidden size for memory efficiency
    
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    # Use mini-batch training with neighborhood sampling
    batch_size = determine_optimal_batch_size(graph, memory_limit_mb)
    loader = NeighborSampler(data.edge_index, node_idx=None,
                           sizes=[10, 10], batch_size=batch_size,
                           shuffle=True, num_workers=0)
    
    # Training loop
    model.train()
    for epoch in range(100):
        total_loss = 0
        
        # Enable garbage collection during training
        gc.enable()
        
        for batch_idx, (batch_size, n_id, adjs) in enumerate(loader):
            # Get edge indices and convert to device
            adjs = [adj.to(device) for adj in adjs]
            
            # Get node features for this mini-batch
            batch_features = {node_type: feat[n_id].to(device) 
                            for node_type, feat in data.items() 
                            if node_type.endswith('_x')}
            
            # Forward pass - with memory tracking
            torch.cuda.reset_peak_memory_stats()
            optimizer.zero_grad()
            
            out = model(batch_features, adjs)
            
            # Get labels for this batch
            batch_labels = labels[n_id[:batch_size]].to(device)
            loss = F.mse_loss(out, batch_labels)
            
            # Backward pass with gradient accumulation for larger effective batch size
            loss.backward()
            
            # Gradient clipping to prevent memory spikes
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            # Check memory usage and reduce batch size if needed
            peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
            if peak_memory > memory_limit_mb * 0.9:
                # Reduce batch size for next iteration
                loader.batch_size = max(1, loader.batch_size // 2)
                print(f"Reduced batch size to {loader.batch_size} due to memory pressure")
            
            # Explicitly clear unnecessary tensors
            del adjs, batch_features, out, batch_labels, loss
            torch.cuda.empty_cache()
            gc.collect()
        
        print(f'Epoch {epoch}, Loss: {total_loss / len(loader)}')
    
    return model
```

## 13. References

1. Yang Y, et al. Treatment Outcome Prediction for Cancer Patients based on Deep Learning and Multi-modal Data. *Nature Medicine*. 2020.
2. Ahmedt-Aristizabal D, et al. Graph Neural Networks in Healthcare: A Survey. *IEEE Journal of Biomedical and Health Informatics*. 2022.
3. Abramian D, Eklund A. Refining the Precision of Glioma Grading: The Role of Deep Learning and Explainable AI. *Scientific Reports*. 2022.
4. Zhao B, et al. Federated Learning for Healthcare: Systematic Survey and Taxonomic Classification of Research Studies. *Journal of Biomedical Informatics*. 2023.
5. Liu X, et al. Self-supervised Learning of Graph Neural Networks for Treatment Effect Estimation. *KDD*. 2021.
6. Zhang Y, et al. Memory-Efficient Training of Graph Neural Networks with Low-Precision Weights. *ICLR*. 2023.
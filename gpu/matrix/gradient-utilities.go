package matrix

/*
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework Accelerate
#include <stdlib.h>
#include "../../internal/cgo/metal_bridge.h"
*/
import "C"

import (
	"fmt"
	"math"
	"math/rand"
	"sort"
	"sync"
	"time"

	"github.com/tsawler/gometal/tensor"
)

// GradientUtilities provides advanced utilities for gradient manipulation,
// analysis, and optimization

// GradientAnalyzer provides detailed analysis of gradients
type GradientAnalyzer struct {
	enabled        bool
	historySize    int
	gradientNorms  []float32
	gradientStats  map[string]*GradientLayerStats
	anomalyTracker *GradientAnomalyTracker
	mutex          sync.RWMutex
}

// GradientLayerStats tracks statistics for gradients of a specific layer
type GradientLayerStats struct {
	LayerName      string
	ParameterCount int64
	NormHistory    []float32
	MagnitudeStats *StatisticalSummary
	DirectionStats *DirectionalStats
	UpdateHistory  []GradientUpdate
	AnomalyCount   int64
	LastUpdated    time.Time
}

// StatisticalSummary provides statistical summary of gradient values
type StatisticalSummary struct {
	Mean        float32
	Variance    float32
	Min         float32
	Max         float32
	Percentiles map[int]float32 // 25th, 50th, 75th, 90th, 95th, 99th
	Skewness    float32
	Kurtosis    float32
}

// DirectionalStats tracks gradient direction statistics
type DirectionalStats struct {
	CosineSimilarities []float32
	AngleChanges       []float32
	Consistency        float32
	Stability          float32
}

// GradientUpdate represents a single gradient update
type GradientUpdate struct {
	Timestamp time.Time
	Norm      float32
	Direction []float32 // Unit vector
	Magnitude float32
	StepSize  float32
}

// GradientAnomalyTracker detects gradient anomalies
type GradientAnomalyTracker struct {
	enabled              bool
	vanishingThreshold   float32
	explodingThreshold   float32
	instabilityThreshold float32
	anomalies            []GradientAnomaly
	detectionCallbacks   []AnomalyCallback
	mutex                sync.RWMutex
}

// GradientAnomaly represents a detected gradient anomaly
type GradientAnomaly struct {
	Type        GradientAnomalyType
	Timestamp   time.Time
	LayerName   string
	Severity    float32
	Description string
	Metadata    map[string]interface{}
}

// GradientAnomalyType represents different types of gradient anomalies
type GradientAnomalyType int

const (
	VanishingGradient GradientAnomalyType = iota
	ExplodingGradient
	GradientInstability
	GradientOscillation
	GradientPlateau
	GradientSpike
)

// AnomalyCallback is called when an anomaly is detected
type AnomalyCallback func(anomaly GradientAnomaly)

// GradientManipulator provides advanced gradient manipulation operations
type GradientManipulator struct {
	clippingHistory []ClippingEvent
	scalingHistory  []ScalingEvent
	noiseInjector   *GradientNoiseInjector
	smoother        *GradientSmoother
	accumulator     *GradientAccumulator
	mutex           sync.RWMutex
}

// ClippingEvent records a gradient clipping event
type ClippingEvent struct {
	Timestamp      time.Time
	OriginalNorm   float32
	ClippedNorm    float32
	ClipRatio      float32
	LayersAffected int
}

// ScalingEvent records a gradient scaling event
type ScalingEvent struct {
	Timestamp      time.Time
	ScaleFactor    float32
	Reason         string
	LayersAffected int
}

// GradientNoiseInjector adds controlled noise to gradients for regularization
type GradientNoiseInjector struct {
	enabled      bool
	noiseType    NoiseType
	noiseLevel   float32
	schedule     NoiseSchedule
	currentStep  int64
	noiseHistory []NoiseEvent
}

// NoiseType represents different types of gradient noise
type NoiseType int

const (
	GaussianNoise NoiseType = iota
	UniformNoise
	SaltPepperNoise
	DropoutNoise
)

// NoiseSchedule controls how noise level changes over time
type NoiseSchedule struct {
	ScheduleType NoiseScheduleType
	InitialLevel float32
	FinalLevel   float32
	DecaySteps   int64
	DecayRate    float32
}

// NoiseScheduleType represents different noise scheduling strategies
type NoiseScheduleType int

const (
	ConstantNoise NoiseScheduleType = iota
	LinearDecay
	ExponentialDecay
	CosineDecay
)

// NoiseEvent records a noise injection event
type NoiseEvent struct {
	Timestamp      time.Time
	NoiseLevel     float32
	NoiseType      NoiseType
	LayersAffected int
}

// GradientSmoother applies smoothing to gradients
type GradientSmoother struct {
	enabled       bool
	smoothingType SmoothingType
	windowSize    int
	alpha         float32 // For EMA
	history       map[string]*SmoothingBuffer
	mutex         sync.RWMutex
}

// SmoothingType represents different gradient smoothing methods
type SmoothingType int

const (
	NoSmoothing SmoothingType = iota
	MovingAverage
	ExponentialMovingAverage
	MedianFilter
	GaussianFilter
)

// SmoothingBuffer maintains smoothing history for a parameter
type SmoothingBuffer struct {
	values   [][]float32
	emaValue []float32
	index    int
	filled   bool
}

// GradientAccumulator handles sophisticated gradient accumulation
type GradientAccumulator struct {
	enabled           bool
	accumulationType  AccumulationType
	accumulationSteps int
	currentStep       int
	accumulatedGrads  map[string]*AccumulationBuffer
	adaptiveThreshold float32
	priorityWeights   map[string]float32
	mutex             sync.RWMutex
}

// AccumulationType represents different accumulation strategies
type AccumulationType int

const (
	SimpleAccumulation AccumulationType = iota
	WeightedAccumulation
	AdaptiveAccumulation
	PriorityAccumulation
)

// AccumulationBuffer stores accumulated gradients
type AccumulationBuffer struct {
	gradients    []*tensor.Tensor
	weights      []float32
	priorities   []float32
	timestamps   []time.Time
	currentIndex int
	filled       bool
}

// Global gradient utilities
var globalGradientAnalyzer *GradientAnalyzer
var globalGradientManipulator *GradientManipulator

func init() {
	globalGradientAnalyzer = NewGradientAnalyzer(100)
	globalGradientManipulator = NewGradientManipulator()
}

// NewGradientAnalyzer creates a new gradient analyzer
func NewGradientAnalyzer(historySize int) *GradientAnalyzer {
	return &GradientAnalyzer{
		enabled:       false,
		historySize:   historySize,
		gradientNorms: make([]float32, 0, historySize),
		gradientStats: make(map[string]*GradientLayerStats),
		anomalyTracker: &GradientAnomalyTracker{
			enabled:              false,
			vanishingThreshold:   1e-7,
			explodingThreshold:   100.0,
			instabilityThreshold: 0.1,
			anomalies:            make([]GradientAnomaly, 0),
			detectionCallbacks:   make([]AnomalyCallback, 0),
		},
	}
}

// EnableGradientAnalysis enables gradient analysis
func EnableGradientAnalysis() {
	globalGradientAnalyzer.mutex.Lock()
	defer globalGradientAnalyzer.mutex.Unlock()
	globalGradientAnalyzer.enabled = true
	globalGradientAnalyzer.anomalyTracker.enabled = true
}

// DisableGradientAnalysis disables gradient analysis
func DisableGradientAnalysis() {
	globalGradientAnalyzer.mutex.Lock()
	defer globalGradientAnalyzer.mutex.Unlock()
	globalGradientAnalyzer.enabled = false
	globalGradientAnalyzer.anomalyTracker.enabled = false
}

// AnalyzeGradients performs comprehensive gradient analysis
func AnalyzeGradients(gradients []*GradientTensor, layerNames []string) (*GradientAnalysisReport, error) {
	if !globalGradientAnalyzer.enabled {
		return nil, nil // Analysis disabled
	}

	if len(gradients) != len(layerNames) {
		return nil, fmt.Errorf("number of gradients (%d) must match number of layer names (%d)", len(gradients), len(layerNames))
	}

	analyzer := globalGradientAnalyzer
	analyzer.mutex.Lock()
	defer analyzer.mutex.Unlock()

	report := &GradientAnalysisReport{
		Timestamp:    time.Now(),
		LayerReports: make(map[string]*LayerGradientReport),
		GlobalStats:  &GlobalGradientStats{},
		Anomalies:    make([]GradientAnomaly, 0),
	}

	var globalNorm float32
	totalParams := int64(0)

	for i, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		layerName := layerNames[i]

		// Analyze individual layer
		layerReport, err := analyzer.analyzeLayerGradients(gt, layerName)
		if err != nil {
			return nil, fmt.Errorf("failed to analyze layer %s: %w", layerName, err)
		}

		report.LayerReports[layerName] = layerReport
		globalNorm += layerReport.Norm * layerReport.Norm
		totalParams += layerReport.ParameterCount

		// Detect anomalies
		anomalies := analyzer.anomalyTracker.detectAnomalies(gt, layerName, layerReport)
		report.Anomalies = append(report.Anomalies, anomalies...)
	}

	// Calculate global statistics
	globalNorm = float32(math.Sqrt(float64(globalNorm)))
	report.GlobalStats.GlobalNorm = globalNorm
	report.GlobalStats.TotalParameters = totalParams
	report.GlobalStats.ActiveLayers = len(report.LayerReports)

	// Update global history
	analyzer.gradientNorms = append(analyzer.gradientNorms, globalNorm)
	if len(analyzer.gradientNorms) > analyzer.historySize {
		analyzer.gradientNorms = analyzer.gradientNorms[1:]
	}

	// Calculate trend
	report.GlobalStats.NormTrend = analyzer.calculateNormTrend()

	return report, nil
}

// GradientAnalysisReport contains comprehensive gradient analysis results
type GradientAnalysisReport struct {
	Timestamp    time.Time
	LayerReports map[string]*LayerGradientReport
	GlobalStats  *GlobalGradientStats
	Anomalies    []GradientAnomaly
}

// LayerGradientReport contains analysis for a single layer
type LayerGradientReport struct {
	LayerName       string
	ParameterCount  int64
	Norm            float32
	Mean            float32
	Variance        float32
	Min             float32
	Max             float32
	Percentiles     map[int]float32
	Sparsity        float32
	DirectionChange float32
	UpdateStability float32
}

// GlobalGradientStats contains global gradient statistics
type GlobalGradientStats struct {
	GlobalNorm      float32
	TotalParameters int64
	ActiveLayers    int
	NormTrend       float32
	HealthScore     float32
}

// analyzeLayerGradients analyzes gradients for a single layer
func (ga *GradientAnalyzer) analyzeLayerGradients(gt *GradientTensor, layerName string) (*LayerGradientReport, error) {
	if err := gt.Gradient.RetrieveCPU(); err != nil {
		return nil, fmt.Errorf("failed to retrieve gradients: %w", err)
	}

	data := gt.Gradient.Data
	report := &LayerGradientReport{
		LayerName:      layerName,
		ParameterCount: int64(len(data)),
		Percentiles:    make(map[int]float32),
	}

	if len(data) == 0 {
		return report, nil
	}

	// Calculate basic statistics
	sum := float32(0)
	sumSquares := float32(0)
	min := data[0]
	max := data[0]
	nonZeroCount := 0

	for _, val := range data {
		sum += val
		sumSquares += val * val

		if val < min {
			min = val
		}
		if val > max {
			max = val
		}
		if val != 0 {
			nonZeroCount++
		}
	}

	n := float32(len(data))
	report.Mean = sum / n
	report.Variance = (sumSquares / n) - (report.Mean * report.Mean)
	report.Min = min
	report.Max = max
	report.Norm = float32(math.Sqrt(float64(sumSquares)))
	report.Sparsity = 1.0 - float32(nonZeroCount)/n

	// Calculate percentiles
	sortedData := make([]float32, len(data))
	copy(sortedData, data)
	sort.Slice(sortedData, func(i, j int) bool {
		return sortedData[i] < sortedData[j]
	})

	percentiles := []int{25, 50, 75, 90, 95, 99}
	for _, p := range percentiles {
		index := int(float32(p) / 100.0 * float32(len(sortedData)-1))
		report.Percentiles[p] = sortedData[index]
	}

	// Update layer statistics
	stats, exists := ga.gradientStats[layerName]
	if !exists {
		stats = &GradientLayerStats{
			LayerName:      layerName,
			ParameterCount: report.ParameterCount,
			NormHistory:    make([]float32, 0, ga.historySize),
			UpdateHistory:  make([]GradientUpdate, 0, ga.historySize),
		}
		ga.gradientStats[layerName] = stats
	}

	// Update history
	stats.NormHistory = append(stats.NormHistory, report.Norm)
	if len(stats.NormHistory) > ga.historySize {
		stats.NormHistory = stats.NormHistory[1:]
	}

	// Calculate direction change if we have history
	if len(stats.UpdateHistory) > 0 {
		lastUpdate := stats.UpdateHistory[len(stats.UpdateHistory)-1]
		report.DirectionChange = ga.calculateDirectionChange(data, lastUpdate.Direction)
	}

	// Record this update
	direction := ga.calculateUnitVector(data)
	update := GradientUpdate{
		Timestamp: time.Now(),
		Norm:      report.Norm,
		Direction: direction,
		Magnitude: report.Norm,
	}

	stats.UpdateHistory = append(stats.UpdateHistory, update)
	if len(stats.UpdateHistory) > ga.historySize {
		stats.UpdateHistory = stats.UpdateHistory[1:]
	}

	stats.LastUpdated = time.Now()

	// Calculate update stability
	report.UpdateStability = ga.calculateUpdateStability(stats.NormHistory)

	return report, nil
}

// calculateUnitVector calculates the unit vector for a gradient
func (ga *GradientAnalyzer) calculateUnitVector(data []float32) []float32 {
	norm := float32(0)
	for _, val := range data {
		norm += val * val
	}
	norm = float32(math.Sqrt(float64(norm)))

	if norm == 0 {
		return make([]float32, len(data))
	}

	unit := make([]float32, len(data))
	for i, val := range data {
		unit[i] = val / norm
	}

	return unit
}

// calculateDirectionChange calculates the change in gradient direction
func (ga *GradientAnalyzer) calculateDirectionChange(current []float32, previous []float32) float32 {
	if len(current) != len(previous) {
		return 0
	}

	currentUnit := ga.calculateUnitVector(current)

	// Calculate cosine similarity
	dotProduct := float32(0)
	for i := range currentUnit {
		dotProduct += currentUnit[i] * previous[i]
	}

	// Clamp to [-1, 1] to handle numerical errors
	if dotProduct > 1 {
		dotProduct = 1
	}
	if dotProduct < -1 {
		dotProduct = -1
	}

	// Return angle in radians
	return float32(math.Acos(float64(dotProduct)))
}

// calculateUpdateStability calculates the stability of gradient updates
func (ga *GradientAnalyzer) calculateUpdateStability(norms []float32) float32 {
	if len(norms) < 2 {
		return 1.0 // Perfect stability with insufficient data
	}

	// Calculate coefficient of variation
	sum := float32(0)
	for _, norm := range norms {
		sum += norm
	}
	mean := sum / float32(len(norms))

	if mean == 0 {
		return 0 // Undefined
	}

	variance := float32(0)
	for _, norm := range norms {
		diff := norm - mean
		variance += diff * diff
	}
	variance /= float32(len(norms))

	stdDev := float32(math.Sqrt(float64(variance)))
	cv := stdDev / mean

	// Return stability as 1 / (1 + cv)
	return 1.0 / (1.0 + cv)
}

// calculateNormTrend calculates the trend in gradient norms
func (ga *GradientAnalyzer) calculateNormTrend() float32 {
	if len(ga.gradientNorms) < 10 {
		return 0 // Insufficient data
	}

	// Simple linear regression to calculate trend
	n := float32(len(ga.gradientNorms))
	sumX := float32(0)
	sumY := float32(0)
	sumXY := float32(0)
	sumX2 := float32(0)

	for i, y := range ga.gradientNorms {
		x := float32(i)
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
	}

	slope := (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX)
	return slope
}

// detectAnomalies detects gradient anomalies
func (gat *GradientAnomalyTracker) detectAnomalies(gt *GradientTensor, layerName string, report *LayerGradientReport) []GradientAnomaly {
	if !gat.enabled {
		return nil
	}

	var anomalies []GradientAnomaly

	// Check for vanishing gradients
	if report.Norm < gat.vanishingThreshold {
		anomaly := GradientAnomaly{
			Type:        VanishingGradient,
			Timestamp:   time.Now(),
			LayerName:   layerName,
			Severity:    gat.vanishingThreshold / (report.Norm + 1e-10),
			Description: fmt.Sprintf("Gradient norm %.2e is below vanishing threshold %.2e", report.Norm, gat.vanishingThreshold),
			Metadata: map[string]interface{}{
				"norm":      report.Norm,
				"threshold": gat.vanishingThreshold,
			},
		}
		anomalies = append(anomalies, anomaly)
	}

	// Check for exploding gradients
	if report.Norm > gat.explodingThreshold {
		anomaly := GradientAnomaly{
			Type:        ExplodingGradient,
			Timestamp:   time.Now(),
			LayerName:   layerName,
			Severity:    report.Norm / gat.explodingThreshold,
			Description: fmt.Sprintf("Gradient norm %.2e exceeds exploding threshold %.2e", report.Norm, gat.explodingThreshold),
			Metadata: map[string]interface{}{
				"norm":      report.Norm,
				"threshold": gat.explodingThreshold,
			},
		}
		anomalies = append(anomalies, anomaly)
	}

	// Check for gradient instability
	if report.UpdateStability < gat.instabilityThreshold {
		anomaly := GradientAnomaly{
			Type:        GradientInstability,
			Timestamp:   time.Now(),
			LayerName:   layerName,
			Severity:    gat.instabilityThreshold / (report.UpdateStability + 1e-10),
			Description: fmt.Sprintf("Gradient stability %.4f is below threshold %.4f", report.UpdateStability, gat.instabilityThreshold),
			Metadata: map[string]interface{}{
				"stability": report.UpdateStability,
				"threshold": gat.instabilityThreshold,
			},
		}
		anomalies = append(anomalies, anomaly)
	}

	// Store anomalies
	gat.anomalies = append(gat.anomalies, anomalies...)

	// Call callbacks
	for _, anomaly := range anomalies {
		for _, callback := range gat.detectionCallbacks {
			callback(anomaly)
		}
	}

	return anomalies
}

// RegisterAnomalyCallback registers a callback for anomaly detection
func RegisterAnomalyCallback(callback AnomalyCallback) {
	globalGradientAnalyzer.anomalyTracker.mutex.Lock()
	defer globalGradientAnalyzer.anomalyTracker.mutex.Unlock()
	globalGradientAnalyzer.anomalyTracker.detectionCallbacks = append(
		globalGradientAnalyzer.anomalyTracker.detectionCallbacks, callback)
}

// NewGradientManipulator creates a new gradient manipulator
func NewGradientManipulator() *GradientManipulator {
	return &GradientManipulator{
		clippingHistory: make([]ClippingEvent, 0),
		scalingHistory:  make([]ScalingEvent, 0),
		noiseInjector: &GradientNoiseInjector{
			enabled:      false,
			noiseType:    GaussianNoise,
			noiseLevel:   0.01,
			noiseHistory: make([]NoiseEvent, 0),
		},
		smoother: &GradientSmoother{
			enabled:       false,
			smoothingType: ExponentialMovingAverage,
			windowSize:    5,
			alpha:         0.9,
			history:       make(map[string]*SmoothingBuffer),
		},
		accumulator: &GradientAccumulator{
			enabled:           false,
			accumulationType:  SimpleAccumulation,
			accumulationSteps: 4,
			accumulatedGrads:  make(map[string]*AccumulationBuffer),
			adaptiveThreshold: 0.1,
			priorityWeights:   make(map[string]float32),
		},
	}
}

// ClipGradientsAdaptive performs adaptive gradient clipping
func ClipGradientsAdaptive(gradients []*GradientTensor, targetNorm float32, adaptationRate float32) (float32, error) {
	manipulator := globalGradientManipulator
	manipulator.mutex.Lock()
	defer manipulator.mutex.Unlock()

	// Calculate current global norm
	globalNorm := float32(0)
	for _, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		if err := gt.Gradient.RetrieveCPU(); err != nil {
			return 0, fmt.Errorf("failed to retrieve gradient: %w", err)
		}

		for _, val := range gt.Gradient.Data {
			globalNorm += val * val
		}
	}
	globalNorm = float32(math.Sqrt(float64(globalNorm)))

	// Adaptive clipping threshold
	var clipThreshold float32
	if len(manipulator.clippingHistory) > 0 {
		lastEvent := manipulator.clippingHistory[len(manipulator.clippingHistory)-1]
		clipThreshold = lastEvent.ClippedNorm + adaptationRate*(targetNorm-lastEvent.ClippedNorm)
	} else {
		clipThreshold = targetNorm
	}

	clippedNorm := globalNorm
	layersAffected := 0

	if globalNorm > clipThreshold {
		scaleFactor := clipThreshold / globalNorm
		clippedNorm = clipThreshold

		// Apply clipping
		for _, gt := range gradients {
			if gt.Gradient == nil {
				continue
			}

			hasNonZero := false
			for i, val := range gt.Gradient.Data {
				newVal := val * scaleFactor
				if newVal != val {
					hasNonZero = true
				}
				gt.Gradient.Data[i] = newVal
			}

			if hasNonZero {
				layersAffected++
			}
		}
	}

	// Record clipping event
	event := ClippingEvent{
		Timestamp:      time.Now(),
		OriginalNorm:   globalNorm,
		ClippedNorm:    clippedNorm,
		ClipRatio:      clippedNorm / globalNorm,
		LayersAffected: layersAffected,
	}

	manipulator.clippingHistory = append(manipulator.clippingHistory, event)

	return clippedNorm, nil
}

// InjectGradientNoise injects controlled noise into gradients
func InjectGradientNoise(gradients []*GradientTensor, noiseConfig *NoiseSchedule, step int64) error {
	manipulator := globalGradientManipulator
	injector := manipulator.noiseInjector

	if !injector.enabled {
		return nil
	}

	manipulator.mutex.Lock()
	defer manipulator.mutex.Unlock()

	// Calculate current noise level based on schedule
	noiseLevel := calculateNoiseLevel(noiseConfig, step)

	layersAffected := 0

	for _, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		if err := gt.Gradient.RetrieveCPU(); err != nil {
			return fmt.Errorf("failed to retrieve gradient: %w", err)
		}

		err := injectNoise(gt.Gradient.Data, injector.noiseType, noiseLevel)
		if err != nil {
			return fmt.Errorf("failed to inject noise: %w", err)
		}

		layersAffected++
	}

	// Record noise injection event
	event := NoiseEvent{
		Timestamp:      time.Now(),
		NoiseLevel:     noiseLevel,
		NoiseType:      injector.noiseType,
		LayersAffected: layersAffected,
	}

	injector.noiseHistory = append(injector.noiseHistory, event)
	injector.currentStep = step

	return nil
}

// calculateNoiseLevel calculates the current noise level based on schedule
func calculateNoiseLevel(schedule *NoiseSchedule, step int64) float32 {
	switch schedule.ScheduleType {
	case ConstantNoise:
		return schedule.InitialLevel

	case LinearDecay:
		if step >= schedule.DecaySteps {
			return schedule.FinalLevel
		}
		progress := float32(step) / float32(schedule.DecaySteps)
		return schedule.InitialLevel + progress*(schedule.FinalLevel-schedule.InitialLevel)

	case ExponentialDecay:
		decayFactor := float32(math.Pow(float64(schedule.DecayRate), float64(step)/float64(schedule.DecaySteps)))
		return schedule.InitialLevel * decayFactor

	case CosineDecay:
		if step >= schedule.DecaySteps {
			return schedule.FinalLevel
		}
		progress := float32(step) / float32(schedule.DecaySteps)
		cosineDecay := 0.5 * (1 + float32(math.Cos(math.Pi*float64(progress))))
		return schedule.FinalLevel + (schedule.InitialLevel-schedule.FinalLevel)*cosineDecay

	default:
		return schedule.InitialLevel
	}
}

// injectNoise injects noise into gradient data
func injectNoise(data []float32, noiseType NoiseType, level float32) error {
	switch noiseType {
	case GaussianNoise:
		for i := range data {
			noise := generateGaussianNoise() * level
			data[i] += noise
		}

	case UniformNoise:
		for i := range data {
			noise := (rand.Float32()*2 - 1) * level
			data[i] += noise
		}

	case SaltPepperNoise:
		for i := range data {
			if rand.Float32() < level {
				if rand.Float32() < 0.5 {
					data[i] = 0 // Pepper noise
				} else {
					data[i] += level // Salt noise
				}
			}
		}

	case DropoutNoise:
		for i := range data {
			if rand.Float32() < level {
				data[i] = 0
			}
		}

	default:
		return fmt.Errorf("unsupported noise type: %v", noiseType)
	}

	return nil
}

// generateGaussianNoise generates Gaussian noise using Box-Muller transform
func generateGaussianNoise() float32 {
	// Simple Box-Muller transform
	u1 := rand.Float32()
	u2 := rand.Float32()

	z := float32(math.Sqrt(-2*math.Log(float64(u1))) * math.Cos(2*math.Pi*float64(u2)))
	return z
}

// SmoothGradients applies smoothing to gradients
func SmoothGradients(gradients []*GradientTensor, layerNames []string) error {
	manipulator := globalGradientManipulator
	smoother := manipulator.smoother

	if !smoother.enabled {
		return nil
	}

	smoother.mutex.Lock()
	defer smoother.mutex.Unlock()

	for i, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		layerName := ""
		if i < len(layerNames) {
			layerName = layerNames[i]
		} else {
			layerName = fmt.Sprintf("layer_%d", i)
		}

		if err := gt.Gradient.RetrieveCPU(); err != nil {
			return fmt.Errorf("failed to retrieve gradient for layer %s: %w", layerName, err)
		}

		err := smoother.smoothLayerGradients(gt.Gradient.Data, layerName)
		if err != nil {
			return fmt.Errorf("failed to smooth gradients for layer %s: %w", layerName, err)
		}
	}

	return nil
}

// smoothLayerGradients applies smoothing to a single layer's gradients
func (gs *GradientSmoother) smoothLayerGradients(data []float32, layerName string) error {
	buffer, exists := gs.history[layerName]
	if !exists {
		buffer = &SmoothingBuffer{
			values:   make([][]float32, gs.windowSize),
			emaValue: make([]float32, len(data)),
			index:    0,
			filled:   false,
		}

		// Initialize with zeros
		for i := range buffer.values {
			buffer.values[i] = make([]float32, len(data))
		}
		copy(buffer.emaValue, data) // Initialize EMA with first value

		gs.history[layerName] = buffer
	}

	switch gs.smoothingType {
	case MovingAverage:
		return gs.applyMovingAverage(data, buffer)

	case ExponentialMovingAverage:
		return gs.applyEMA(data, buffer)

	case MedianFilter:
		return gs.applyMedianFilter(data, buffer)

	case GaussianFilter:
		return gs.applyGaussianFilter(data, buffer)

	default:
		return nil // No smoothing
	}
}

// applyMovingAverage applies moving average smoothing
func (gs *GradientSmoother) applyMovingAverage(data []float32, buffer *SmoothingBuffer) error {
	// Store current values
	copy(buffer.values[buffer.index], data)
	buffer.index = (buffer.index + 1) % gs.windowSize

	if !buffer.filled && buffer.index == 0 {
		buffer.filled = true
	}

	// Calculate moving average
	count := gs.windowSize
	if !buffer.filled {
		count = buffer.index
		if count == 0 {
			count = 1
		}
	}

	for i := range data {
		sum := float32(0)
		for j := 0; j < count; j++ {
			sum += buffer.values[j][i]
		}
		data[i] = sum / float32(count)
	}

	return nil
}

// applyEMA applies exponential moving average smoothing
func (gs *GradientSmoother) applyEMA(data []float32, buffer *SmoothingBuffer) error {
	for i := range data {
		buffer.emaValue[i] = gs.alpha*buffer.emaValue[i] + (1-gs.alpha)*data[i]
		data[i] = buffer.emaValue[i]
	}
	return nil
}

// applyMedianFilter applies median filter smoothing
func (gs *GradientSmoother) applyMedianFilter(data []float32, buffer *SmoothingBuffer) error {
	// Store current values
	copy(buffer.values[buffer.index], data)
	buffer.index = (buffer.index + 1) % gs.windowSize

	if !buffer.filled && buffer.index == 0 {
		buffer.filled = true
	}

	count := gs.windowSize
	if !buffer.filled {
		count = buffer.index + 1
	}

	// Calculate median for each parameter
	for i := range data {
		values := make([]float32, count)
		for j := 0; j < count; j++ {
			values[j] = buffer.values[j][i]
		}

		sort.Slice(values, func(a, b int) bool {
			return values[a] < values[b]
		})

		if count%2 == 0 {
			data[i] = (values[count/2-1] + values[count/2]) / 2
		} else {
			data[i] = values[count/2]
		}
	}

	return nil
}

// applyGaussianFilter applies Gaussian filter smoothing
func (gs *GradientSmoother) applyGaussianFilter(data []float32, buffer *SmoothingBuffer) error {
	// Generate Gaussian weights
	sigma := float32(gs.windowSize) / 6.0 // 3-sigma rule
	weights := make([]float32, gs.windowSize)
	weightSum := float32(0)

	center := gs.windowSize / 2
	for i := 0; i < gs.windowSize; i++ {
		diff := float32(i - center)
		weights[i] = float32(math.Exp(-float64(diff*diff) / (2 * float64(sigma*sigma))))
		weightSum += weights[i]
	}

	// Normalize weights
	for i := range weights {
		weights[i] /= weightSum
	}

	// Store current values
	copy(buffer.values[buffer.index], data)
	buffer.index = (buffer.index + 1) % gs.windowSize

	if !buffer.filled && buffer.index == 0 {
		buffer.filled = true
	}

	count := gs.windowSize
	if !buffer.filled {
		count = buffer.index + 1
	}

	// Apply Gaussian filter
	for i := range data {
		weightedSum := float32(0)
		usedWeightSum := float32(0)

		for j := 0; j < count; j++ {
			weightedSum += weights[j] * buffer.values[j][i]
			usedWeightSum += weights[j]
		}

		if usedWeightSum > 0 {
			data[i] = weightedSum / usedWeightSum
		}
	}

	return nil
}

// AccumulateGradientsAdvanced performs advanced gradient accumulation
func AccumulateGradientsAdvanced(gradients []*GradientTensor, layerNames []string, weights []float32) error {
	manipulator := globalGradientManipulator
	accumulator := manipulator.accumulator

	if !accumulator.enabled {
		return nil
	}

	accumulator.mutex.Lock()
	defer accumulator.mutex.Unlock()

	for i, gt := range gradients {
		if gt.Gradient == nil {
			continue
		}

		layerName := ""
		if i < len(layerNames) {
			layerName = layerNames[i]
		} else {
			layerName = fmt.Sprintf("layer_%d", i)
		}

		weight := float32(1.0)
		if i < len(weights) {
			weight = weights[i]
		}

		if err := gt.Gradient.RetrieveCPU(); err != nil {
			return fmt.Errorf("failed to retrieve gradient for layer %s: %w", layerName, err)
		}

		err := accumulator.accumulateLayerGradients(gt.Gradient, layerName, weight)
		if err != nil {
			return fmt.Errorf("failed to accumulate gradients for layer %s: %w", layerName, err)
		}
	}

	accumulator.currentStep++
	return nil
}

// accumulateLayerGradients accumulates gradients for a single layer
func (ga *GradientAccumulator) accumulateLayerGradients(grad *tensor.Tensor, layerName string, weight float32) error {
	buffer, exists := ga.accumulatedGrads[layerName]
	if !exists {
		buffer = &AccumulationBuffer{
			gradients:    make([]*tensor.Tensor, ga.accumulationSteps),
			weights:      make([]float32, ga.accumulationSteps),
			priorities:   make([]float32, ga.accumulationSteps),
			timestamps:   make([]time.Time, ga.accumulationSteps),
			currentIndex: 0,
			filled:       false,
		}
		ga.accumulatedGrads[layerName] = buffer
	}

	// Create a copy of the gradient
	gradData := make([]float32, len(grad.Data))
	copy(gradData, grad.Data)

	gradCopy, err := tensor.NewTensor(grad.Shape, gradData)
	if err != nil {
		return fmt.Errorf("failed to create gradient copy: %w", err)
	}

	// Store in buffer
	buffer.gradients[buffer.currentIndex] = gradCopy
	buffer.weights[buffer.currentIndex] = weight
	buffer.timestamps[buffer.currentIndex] = time.Now()

	// Calculate priority based on accumulation type
	switch ga.accumulationType {
	case SimpleAccumulation:
		buffer.priorities[buffer.currentIndex] = 1.0

	case WeightedAccumulation:
		buffer.priorities[buffer.currentIndex] = weight

	case AdaptiveAccumulation:
		// Priority based on gradient magnitude
		norm := float32(0)
		for _, val := range grad.Data {
			norm += val * val
		}
		norm = float32(math.Sqrt(float64(norm)))
		buffer.priorities[buffer.currentIndex] = norm

	case PriorityAccumulation:
		// Use pre-defined priority weights
		if priority, exists := ga.priorityWeights[layerName]; exists {
			buffer.priorities[buffer.currentIndex] = priority
		} else {
			buffer.priorities[buffer.currentIndex] = 1.0
		}
	}

	buffer.currentIndex = (buffer.currentIndex + 1) % ga.accumulationSteps
	if !buffer.filled && buffer.currentIndex == 0 {
		buffer.filled = true
	}

	// Apply accumulated gradient to original
	return ga.applyAccumulatedGradient(grad, buffer)
}

// applyAccumulatedGradient applies the accumulated gradient
func (ga *GradientAccumulator) applyAccumulatedGradient(grad *tensor.Tensor, buffer *AccumulationBuffer) error {
	count := ga.accumulationSteps
	if !buffer.filled {
		count = buffer.currentIndex
		if count == 0 {
			return nil // No gradients to accumulate yet
		}
	}

	// Calculate weighted average
	totalWeight := float32(0)
	for i := 0; i < count; i++ {
		totalWeight += buffer.weights[i] * buffer.priorities[i]
	}

	if totalWeight == 0 {
		return nil
	}

	// Zero out the gradient
	for i := range grad.Data {
		grad.Data[i] = 0
	}

	// Accumulate weighted gradients
	for i := 0; i < count; i++ {
		weight := buffer.weights[i] * buffer.priorities[i] / totalWeight

		if err := buffer.gradients[i].RetrieveCPU(); err != nil {
			continue // Skip on error
		}

		for j, val := range buffer.gradients[i].Data {
			grad.Data[j] += val * weight
		}
	}

	return nil
}

// EnableGradientSmoothing enables gradient smoothing
func EnableGradientSmoothing(smoothingType SmoothingType, windowSize int, alpha float32) {
	globalGradientManipulator.smoother.mutex.Lock()
	defer globalGradientManipulator.smoother.mutex.Unlock()

	globalGradientManipulator.smoother.enabled = true
	globalGradientManipulator.smoother.smoothingType = smoothingType
	globalGradientManipulator.smoother.windowSize = windowSize
	globalGradientManipulator.smoother.alpha = alpha
}

// DisableGradientSmoothing disables gradient smoothing
func DisableGradientSmoothing() {
	globalGradientManipulator.smoother.mutex.Lock()
	defer globalGradientManipulator.smoother.mutex.Unlock()
	globalGradientManipulator.smoother.enabled = false
}

// EnableGradientNoise enables gradient noise injection
func EnableGradientNoise(noiseType NoiseType, schedule *NoiseSchedule) {
	globalGradientManipulator.noiseInjector.enabled = true
	globalGradientManipulator.noiseInjector.noiseType = noiseType
	globalGradientManipulator.noiseInjector.schedule = *schedule
	globalGradientManipulator.noiseInjector.currentStep = 0
}

// DisableGradientNoise disables gradient noise injection
func DisableGradientNoise() {
	globalGradientManipulator.noiseInjector.enabled = false
}

// EnableAdvancedGradientAccumulation enables advanced gradient accumulation
func EnableAdvancedGradientAccumulation(accumulationType AccumulationType, steps int, threshold float32) {
	accumulator := globalGradientManipulator.accumulator
	accumulator.mutex.Lock()
	defer accumulator.mutex.Unlock()

	accumulator.enabled = true
	accumulator.accumulationType = accumulationType
	accumulator.accumulationSteps = steps
	accumulator.adaptiveThreshold = threshold
	accumulator.currentStep = 0
}

// DisableAdvancedGradientAccumulation disables advanced gradient accumulation
func DisableAdvancedGradientAccumulation() {
	globalGradientManipulator.accumulator.mutex.Lock()
	defer globalGradientManipulator.accumulator.mutex.Unlock()
	globalGradientManipulator.accumulator.enabled = false
}

// SetLayerPriority sets the priority weight for a specific layer
func SetLayerPriority(layerName string, priority float32) {
	globalGradientManipulator.accumulator.mutex.Lock()
	defer globalGradientManipulator.accumulator.mutex.Unlock()
	globalGradientManipulator.accumulator.priorityWeights[layerName] = priority
}

// GetGradientUtilitiesStats returns comprehensive statistics
func GetGradientUtilitiesStats() map[string]interface{} {
	stats := make(map[string]interface{})

	// Analyzer stats
	analyzer := globalGradientAnalyzer
	analyzer.mutex.RLock()
	stats["analyzer_enabled"] = analyzer.enabled
	stats["gradient_history_size"] = len(analyzer.gradientNorms)
	stats["tracked_layers"] = len(analyzer.gradientStats)
	stats["anomaly_detection_enabled"] = analyzer.anomalyTracker.enabled
	stats["total_anomalies"] = len(analyzer.anomalyTracker.anomalies)
	analyzer.mutex.RUnlock()

	// Manipulator stats
	manipulator := globalGradientManipulator
	manipulator.mutex.RLock()
	stats["clipping_events"] = len(manipulator.clippingHistory)
	stats["scaling_events"] = len(manipulator.scalingHistory)
	stats["noise_injection_enabled"] = manipulator.noiseInjector.enabled
	stats["smoothing_enabled"] = manipulator.smoother.enabled
	stats["accumulation_enabled"] = manipulator.accumulator.enabled

	if len(manipulator.clippingHistory) > 0 {
		lastClip := manipulator.clippingHistory[len(manipulator.clippingHistory)-1]
		stats["last_clip_ratio"] = lastClip.ClipRatio
		stats["last_clipped_norm"] = lastClip.ClippedNorm
	}

	if manipulator.noiseInjector.enabled {
		stats["noise_type"] = manipulator.noiseInjector.noiseType
		stats["current_noise_step"] = manipulator.noiseInjector.currentStep
		stats["noise_events"] = len(manipulator.noiseInjector.noiseHistory)
	}

	if manipulator.smoother.enabled {
		stats["smoothing_type"] = manipulator.smoother.smoothingType
		stats["smoothing_window_size"] = manipulator.smoother.windowSize
		stats["smoothed_layers"] = len(manipulator.smoother.history)
	}

	if manipulator.accumulator.enabled {
		stats["accumulation_type"] = manipulator.accumulator.accumulationType
		stats["accumulation_steps"] = manipulator.accumulator.accumulationSteps
		stats["accumulated_layers"] = len(manipulator.accumulator.accumulatedGrads)
		stats["current_accumulation_step"] = manipulator.accumulator.currentStep
	}

	manipulator.mutex.RUnlock()

	return stats
}

// PrintGradientUtilitiesStats prints detailed gradient utilities statistics
func PrintGradientUtilitiesStats() {
	stats := GetGradientUtilitiesStats()

	fmt.Println("=== Gradient Utilities Statistics ===")

	// Analyzer stats
	fmt.Printf("Gradient Analysis Enabled: %v\n", stats["analyzer_enabled"])
	if enabled, ok := stats["analyzer_enabled"].(bool); ok && enabled {
		fmt.Printf("  History Size: %v\n", stats["gradient_history_size"])
		fmt.Printf("  Tracked Layers: %v\n", stats["tracked_layers"])
		fmt.Printf("  Anomaly Detection: %v\n", stats["anomaly_detection_enabled"])
		fmt.Printf("  Total Anomalies: %v\n", stats["total_anomalies"])
	}

	// Manipulator stats
	fmt.Printf("\nGradient Manipulation:\n")
	fmt.Printf("  Clipping Events: %v\n", stats["clipping_events"])
	fmt.Printf("  Scaling Events: %v\n", stats["scaling_events"])

	if lastClipRatio, ok := stats["last_clip_ratio"].(float32); ok {
		fmt.Printf("  Last Clip Ratio: %.4f\n", lastClipRatio)
	}

	fmt.Printf("  Noise Injection: %v\n", stats["noise_injection_enabled"])
	if enabled, ok := stats["noise_injection_enabled"].(bool); ok && enabled {
		fmt.Printf("    Noise Type: %v\n", stats["noise_type"])
		fmt.Printf("    Current Step: %v\n", stats["current_noise_step"])
		fmt.Printf("    Noise Events: %v\n", stats["noise_events"])
	}

	fmt.Printf("  Gradient Smoothing: %v\n", stats["smoothing_enabled"])
	if enabled, ok := stats["smoothing_enabled"].(bool); ok && enabled {
		fmt.Printf("    Smoothing Type: %v\n", stats["smoothing_type"])
		fmt.Printf("    Window Size: %v\n", stats["smoothing_window_size"])
		fmt.Printf("    Smoothed Layers: %v\n", stats["smoothed_layers"])
	}

	fmt.Printf("  Advanced Accumulation: %v\n", stats["accumulation_enabled"])
	if enabled, ok := stats["accumulation_enabled"].(bool); ok && enabled {
		fmt.Printf("    Accumulation Type: %v\n", stats["accumulation_type"])
		fmt.Printf("    Accumulation Steps: %v\n", stats["accumulation_steps"])
		fmt.Printf("    Accumulated Layers: %v\n", stats["accumulated_layers"])
		fmt.Printf("    Current Step: %v\n", stats["current_accumulation_step"])
	}
}

// ResetGradientUtilities resets all gradient utilities to their initial state
func ResetGradientUtilities() {
	// Reset analyzer
	globalGradientAnalyzer.mutex.Lock()
	globalGradientAnalyzer.gradientNorms = globalGradientAnalyzer.gradientNorms[:0]
	globalGradientAnalyzer.gradientStats = make(map[string]*GradientLayerStats)
	globalGradientAnalyzer.anomalyTracker.anomalies = globalGradientAnalyzer.anomalyTracker.anomalies[:0]
	globalGradientAnalyzer.mutex.Unlock()

	// Reset manipulator
	globalGradientManipulator.mutex.Lock()
	globalGradientManipulator.clippingHistory = globalGradientManipulator.clippingHistory[:0]
	globalGradientManipulator.scalingHistory = globalGradientManipulator.scalingHistory[:0]
	globalGradientManipulator.noiseInjector.noiseHistory = globalGradientManipulator.noiseInjector.noiseHistory[:0]
	globalGradientManipulator.noiseInjector.currentStep = 0
	globalGradientManipulator.smoother.history = make(map[string]*SmoothingBuffer)
	globalGradientManipulator.accumulator.accumulatedGrads = make(map[string]*AccumulationBuffer)
	globalGradientManipulator.accumulator.currentStep = 0
	globalGradientManipulator.mutex.Unlock()
}

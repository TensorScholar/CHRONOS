//! Natural language explanation generation system for algorithm behaviors
//!
//! This module provides a framework for generating natural language explanations
//! of algorithm behaviors, patterns, and anomalies at multiple levels of expertise.
//! It maintains mathematical precision while adapting explanations to the user's
//! expertise level.
//!
//! Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use thiserror::Error;
use serde::{Serialize, Deserialize};
use log::{debug, info, warn, error};

use crate::algorithm::traits::{Algorithm, AlgorithmState};
use crate::education::progressive::{ExpertiseLevel, ProgressiveDisclosure};
use crate::insights::pattern::{Pattern, PatternMatch};
use crate::insights::anomaly::{Anomaly, AnomalyMatch};
use crate::utils::math::{normalize};

/// Error type for explanation generation
#[derive(Error, Debug)]
pub enum ExplanationError {
    /// Template not found
    #[error("Template not found: {0}")]
    TemplateNotFound(String),
    
    /// Variable not found
    #[error("Variable not found: {0}")]
    VariableNotFound(String),
    
    /// Transformation error
    #[error("Transformation error: {0}")]
    TransformationError(String),
    
    /// Validation error
    #[error("Validation error: {0}")]
    ValidationError(String),
}

/// Result type for explanation operations
pub type ExplanationResult<T> = Result<T, ExplanationError>;

/// Explanation component with multiple complexity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Explanation {
    /// Title of the explanation
    pub title: String,
    
    /// Concise summary of the explanation (1-2 sentences)
    pub summary: String,
    
    /// Structured explanation with different sections
    pub sections: Vec<ExplanationSection>,
    
    /// Expertise level this explanation is targeted at
    pub expertise_level: ExpertiseLevel,
    
    /// Conceptual references for cross-linking
    pub concept_refs: Vec<ConceptReference>,
    
    /// Visual annotations for integration with visualization
    pub visual_annotations: Vec<VisualAnnotation>,
    
    /// Mathematical expressions included in the explanation
    pub math_expressions: Vec<MathExpression>,
    
    /// Code examples included in the explanation
    pub code_examples: Vec<CodeExample>,
    
    /// Metadata about the explanation
    pub metadata: HashMap<String, String>,
}

/// Section of an explanation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExplanationSection {
    /// Title of the section
    pub title: String,
    
    /// Content of the section in markdown format
    pub content: String,
    
    /// Importance of this section (0.0-1.0)
    pub importance: f64,
    
    /// Order of this section within the explanation
    pub order: u32,
}

/// Reference to a conceptual element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptReference {
    /// Name of the concept
    pub name: String,
    
    /// Type of the concept
    pub concept_type: String,
    
    /// Brief description of the concept
    pub description: String,
    
    /// Link to more information
    pub link: Option<String>,
}

/// Visual annotation for visualization integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisualAnnotation {
    /// Target element identifier
    pub target_id: String,
    
    /// Type of annotation
    pub annotation_type: String,
    
    /// Content of the annotation
    pub content: String,
    
    /// Style parameters
    pub style: HashMap<String, String>,
}

/// Mathematical expression with multiple formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MathExpression {
    /// Identifier for the expression
    pub id: String,
    
    /// LaTeX representation
    pub latex: String,
    
    /// MathML representation
    pub mathml: Option<String>,
    
    /// Plain text representation
    pub plain_text: String,
    
    /// Description of the expression
    pub description: String,
}

/// Code example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExample {
    /// Identifier for the code example
    pub id: String,
    
    /// Language of the code
    pub language: String,
    
    /// Code content
    pub code: String,
    
    /// Description of the code
    pub description: String,
}

/// Explanation template for generating explanations
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExplanationTemplate {
    /// Template identifier
    pub id: String,
    
    /// Template for the title
    pub title_template: String,
    
    /// Template for the summary
    pub summary_template: String,
    
    /// Templates for sections
    pub section_templates: Vec<SectionTemplate>,
    
    /// Templates for concept references
    pub concept_templates: Vec<ConceptTemplate>,
    
    /// Templates for visual annotations
    pub annotation_templates: Vec<AnnotationTemplate>,
    
    /// Templates for mathematical expressions
    pub math_templates: Vec<MathTemplate>,
    
    /// Templates for code examples
    pub code_templates: Vec<CodeTemplate>,
    
    /// Template applicability conditions
    pub conditions: Vec<TemplateCondition>,
    
    /// Expertise level adaptations
    pub expertise_adaptations: HashMap<ExpertiseLevel, ExpertiseAdaptation>,
}

/// Template for an explanation section
#[derive(Debug, Clone, Serialize, Deserialize)]
struct SectionTemplate {
    /// Section identifier
    pub id: String,
    
    /// Template for the title
    pub title_template: String,
    
    /// Template for the content
    pub content_template: String,
    
    /// Base importance of this section
    pub base_importance: f64,
    
    /// Base order of this section
    pub base_order: u32,
    
    /// Minimum expertise level to include this section
    pub min_expertise: ExpertiseLevel,
}

/// Template for a concept reference
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ConceptTemplate {
    /// Concept identifier
    pub id: String,
    
    /// Template for the name
    pub name_template: String,
    
    /// Type of the concept
    pub concept_type: String,
    
    /// Template for the description
    pub description_template: String,
    
    /// Template for the link
    pub link_template: Option<String>,
    
    /// Minimum expertise level to include this concept
    pub min_expertise: ExpertiseLevel,
}

/// Template for a visual annotation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct AnnotationTemplate {
    /// Annotation identifier
    pub id: String,
    
    /// Template for the target ID
    pub target_template: String,
    
    /// Type of the annotation
    pub annotation_type: String,
    
    /// Template for the content
    pub content_template: String,
    
    /// Style parameters
    pub style: HashMap<String, String>,
    
    /// Minimum expertise level to include this annotation
    pub min_expertise: ExpertiseLevel,
}

/// Template for a mathematical expression
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MathTemplate {
    /// Expression identifier
    pub id: String,
    
    /// Template for LaTeX representation
    pub latex_template: String,
    
    /// Template for MathML representation
    pub mathml_template: Option<String>,
    
    /// Template for plain text representation
    pub plaintext_template: String,
    
    /// Template for the description
    pub description_template: String,
    
    /// Minimum expertise level to include this expression
    pub min_expertise: ExpertiseLevel,
}

/// Template for a code example
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CodeTemplate {
    /// Example identifier
    pub id: String,
    
    /// Language of the code
    pub language: String,
    
    /// Template for the code
    pub code_template: String,
    
    /// Template for the description
    pub description_template: String,
    
    /// Minimum expertise level to include this example
    pub min_expertise: ExpertiseLevel,
}

/// Condition for template applicability
#[derive(Debug, Clone, Serialize, Deserialize)]
struct TemplateCondition {
    /// Variable name to check
    pub variable: String,
    
    /// Operator for the condition
    pub operator: ConditionOperator,
    
    /// Value to compare against
    pub value: ConditionValue,
}

/// Operator for template conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ConditionOperator {
    /// Equal to
    Equals,
    
    /// Not equal to
    NotEquals,
    
    /// Greater than
    GreaterThan,
    
    /// Less than
    LessThan,
    
    /// Contains
    Contains,
    
    /// Does not contain
    NotContains,
}

/// Value for template conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
enum ConditionValue {
    /// String value
    String(String),
    
    /// Number value
    Number(f64),
    
    /// Boolean value
    Boolean(bool),
    
    /// Reference to another variable
    Variable(String),
}

/// Expertise level adaptation for a template
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ExpertiseAdaptation {
    /// Sections to include at this expertise level
    pub include_sections: Vec<String>,
    
    /// Sections to exclude at this expertise level
    pub exclude_sections: Vec<String>,
    
    /// Concepts to include at this expertise level
    pub include_concepts: Vec<String>,
    
    /// Concepts to exclude at this expertise level
    pub exclude_concepts: Vec<String>,
    
    /// Annotations to include at this expertise level
    pub include_annotations: Vec<String>,
    
    /// Annotations to exclude at this expertise level
    pub exclude_annotations: Vec<String>,
    
    /// Expressions to include at this expertise level
    pub include_expressions: Vec<String>,
    
    /// Expressions to exclude at this expertise level
    pub exclude_expressions: Vec<String>,
    
    /// Examples to include at this expertise level
    pub include_examples: Vec<String>,
    
    /// Examples to exclude at this expertise level
    pub exclude_examples: Vec<String>,
    
    /// Linguistic transformations for this expertise level
    pub linguistic_transformations: Vec<LinguisticTransformation>,
}

/// Linguistic transformation for expertise adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
struct LinguisticTransformation {
    /// Type of transformation
    pub transformation_type: TransformationType,
    
    /// Parameters for the transformation
    pub parameters: HashMap<String, String>,
}

/// Type of linguistic transformation
#[derive(Debug, Clone, Serialize, Deserialize)]
enum TransformationType {
    /// Simplify technical terminology
    SimplifyTerminology,
    
    /// Expand abbreviations
    ExpandAbbreviations,
    
    /// Add examples
    AddExamples,
    
    /// Provide more context
    ProvideContext,
    
    /// Reduce sentence complexity
    ReduceSentenceComplexity,
    
    /// Add prerequisite explanation
    AddPrerequisites,
}

/// Interface for explanation generation
pub trait ExplanationGenerator: Send + Sync {
    /// Generate an explanation for a pattern match
    fn generate_explanation(
        &self,
        pattern_match: &PatternMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation;
    
    /// Generate an explanation for an anomaly match
    fn generate_explanation_for_anomaly(
        &self,
        anomaly_match: &AnomalyMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation;
    
    /// Generate an explanation for an algorithm state
    fn generate_explanation_for_state(
        &self,
        state: &AlgorithmState,
        algorithm: &dyn Algorithm,
        expertise_level: ExpertiseLevel
    ) -> Explanation;
    
    /// Adapt an explanation to a different expertise level
    fn adapt_explanation(
        &self,
        explanation: &Explanation,
        target_expertise: ExpertiseLevel
    ) -> ExplanationResult<Explanation>;
    
    /// Combine multiple explanations
    fn combine_explanations(
        &self,
        explanations: &[Explanation]
    ) -> ExplanationResult<Explanation>;
}

/// Implementation of the explanation generator
pub struct ExplanationGeneratorImpl {
    /// Templates for explanations
    templates: Arc<RwLock<HashMap<String, ExplanationTemplate>>>,
    
    /// Progressive disclosure system
    progressive_disclosure: Arc<ProgressiveDisclosure>,
    
    /// Template engine for rendering templates
    template_engine: TemplateEngine,
    
    /// Linguistic transformer for adapting language
    linguistic_transformer: LinguisticTransformer,
    
    /// Semantic validator for ensuring correctness
    semantic_validator: SemanticValidator,
}

/// Template engine for rendering templates
struct TemplateEngine {
    /// Rendering environment
    environment: HashMap<String, String>,
}

impl TemplateEngine {
    /// Create a new template engine
    fn new() -> Self {
        Self {
            environment: HashMap::new(),
        }
    }
    
    /// Set a variable in the environment
    fn set_variable(&mut self, name: &str, value: &str) {
        self.environment.insert(name.to_string(), value.to_string());
    }
    
    /// Set multiple variables in the environment
    fn set_variables(&mut self, variables: &HashMap<String, String>) {
        for (name, value) in variables {
            self.set_variable(name, value);
        }
    }
    
    /// Render a template with the current environment
    fn render(&self, template: &str) -> ExplanationResult<String> {
        let mut result = template.to_string();
        
        // Simple variable substitution
        for (name, value) in &self.environment {
            let placeholder = format!("{{{}}}", name);
            result = result.replace(&placeholder, value);
        }
        
        // Handle conditional sections
        // Format: {if:var}content{endif}
        let if_regex = regex::Regex::new(r"\{if:([^\}]+)\}(.*?)\{endif\}").unwrap();
        result = if_regex.replace_all(&result, |caps: &regex::Captures| {
            let var_name = &caps[1];
            let content = &caps[2];
            
            if let Some(value) = self.environment.get(var_name) {
                if value == "true" || value == "1" || !value.is_empty() {
                    return content.to_string();
                }
            }
            
            String::new()
        }).to_string();
        
        // Handle negated conditional sections
        // Format: {ifnot:var}content{endif}
        let ifnot_regex = regex::Regex::new(r"\{ifnot:([^\}]+)\}(.*?)\{endif\}").unwrap();
        result = ifnot_regex.replace_all(&result, |caps: &regex::Captures| {
            let var_name = &caps[1];
            let content = &caps[2];
            
            if let Some(value) = self.environment.get(var_name) {
                if value == "false" || value == "0" || value.is_empty() {
                    return content.to_string();
                }
            } else {
                return content.to_string();
            }
            
            String::new()
        }).to_string();
        
        // Handle loops
        // Format: {foreach:var in list}content with {var}{endforeach}
        let foreach_regex = regex::Regex::new(r"\{foreach:([^\s]+)\s+in\s+([^\}]+)\}(.*?)\{endforeach\}").unwrap();
        result = foreach_regex.replace_all(&result, |caps: &regex::Captures| {
            let var_name = &caps[1];
            let list_name = &caps[2];
            let content_template = &caps[3];
            
            if let Some(list_value) = self.environment.get(list_name) {
                let items: Vec<&str> = list_value.split(',').collect();
                let mut expanded = String::new();
                
                for item in items {
                    let mut item_template = content_template.to_string();
                    item_template = item_template.replace(&format!("{{{}}}", var_name), item);
                    expanded.push_str(&item_template);
                }
                
                return expanded;
            }
            
            String::new()
        }).to_string();
        
        Ok(result)
    }
}

/// Linguistic transformer for adapting language
struct LinguisticTransformer {
    /// Terminology simplification mappings
    terminology_mappings: HashMap<String, HashMap<String, String>>,
    
    /// Abbreviation expansions
    abbreviation_expansions: HashMap<String, String>,
    
    /// Example templates
    example_templates: HashMap<String, Vec<String>>,
    
    /// Context templates
    context_templates: HashMap<String, Vec<String>>,
    
    /// Prerequisite explanations
    prerequisite_explanations: HashMap<String, String>,
}

impl LinguisticTransformer {
    /// Create a new linguistic transformer
    fn new() -> Self {
        Self {
            terminology_mappings: HashMap::new(),
            abbreviation_expansions: HashMap::new(),
            example_templates: HashMap::new(),
            context_templates: HashMap::new(),
            prerequisite_explanations: HashMap::new(),
        }
    }
    
    /// Initialize with default transformations
    fn initialize(&mut self) {
        // Initialize terminology mappings
        let mut advanced_to_intermediate = HashMap::new();
        advanced_to_intermediate.insert("O(n log n)".to_string(), "logarithmic time complexity".to_string());
        advanced_to_intermediate.insert("memoization".to_string(), "caching previous results".to_string());
        advanced_to_intermediate.insert("recursive case".to_string(), "recursive step".to_string());
        
        let mut intermediate_to_beginner = HashMap::new();
        intermediate_to_beginner.insert("logarithmic time complexity".to_string(), "efficient processing time".to_string());
        intermediate_to_beginner.insert("caching previous results".to_string(), "remembering previous answers".to_string());
        intermediate_to_beginner.insert("recursive step".to_string(), "self-repeating step".to_string());
        
        self.terminology_mappings.insert("advanced_to_intermediate".to_string(), advanced_to_intermediate);
        self.terminology_mappings.insert("intermediate_to_beginner".to_string(), intermediate_to_beginner);
        
        // Initialize abbreviation expansions
        self.abbreviation_expansions.insert("BFS".to_string(), "Breadth-First Search".to_string());
        self.abbreviation_expansions.insert("DFS".to_string(), "Depth-First Search".to_string());
        self.abbreviation_expansions.insert("DP".to_string(), "Dynamic Programming".to_string());
        
        // Initialize example templates
        self.example_templates.insert(
            "algorithm".to_string(),
            vec![
                "For example, when sorting a list of numbers [5, 2, 8, 1], this algorithm would...".to_string(),
                "To illustrate, if we apply this to finding the shortest path from A to B, it would...".to_string(),
            ]
        );
        
        // Initialize context templates
        self.context_templates.insert(
            "algorithm".to_string(),
            vec![
                "This is important because it affects how quickly the algorithm can solve larger problems.".to_string(),
                "Understanding this helps us see why the algorithm makes certain choices.".to_string(),
            ]
        );
        
        // Initialize prerequisite explanations
        self.prerequisite_explanations.insert(
            "dynamic programming".to_string(),
            "Dynamic programming is an approach that breaks down complex problems into simpler subproblems and stores their solutions to avoid redundant calculations.".to_string(),
        );
    }
    
    /// Apply a transformation to text
    fn apply_transformation(
        &self,
        text: &str,
        transformation: &LinguisticTransformation,
        expertise_level: &ExpertiseLevel
    ) -> ExplanationResult<String> {
        match transformation.transformation_type {
            TransformationType::SimplifyTerminology => {
                self.simplify_terminology(text, expertise_level)
            },
            TransformationType::ExpandAbbreviations => {
                self.expand_abbreviations(text)
            },
            TransformationType::AddExamples => {
                self.add_examples(text, &transformation.parameters)
            },
            TransformationType::ProvideContext => {
                self.provide_context(text, &transformation.parameters)
            },
            TransformationType::ReduceSentenceComplexity => {
                self.reduce_sentence_complexity(text)
            },
            TransformationType::AddPrerequisites => {
                self.add_prerequisites(text, &transformation.parameters)
            },
        }
    }
    
    /// Simplify terminology based on expertise level
    fn simplify_terminology(
        &self,
        text: &str,
        expertise_level: &ExpertiseLevel
    ) -> ExplanationResult<String> {
        let mut result = text.to_string();
        
        match expertise_level {
            ExpertiseLevel::Beginner => {
                // Apply intermediate to beginner mapping
                if let Some(mappings) = self.terminology_mappings.get("intermediate_to_beginner") {
                    for (term, replacement) in mappings {
                        result = result.replace(term, replacement);
                    }
                }
                
                // Also apply advanced to intermediate mapping
                if let Some(mappings) = self.terminology_mappings.get("advanced_to_intermediate") {
                    for (term, replacement) in mappings {
                        result = result.replace(term, replacement);
                    }
                }
            },
            ExpertiseLevel::Intermediate => {
                // Apply advanced to intermediate mapping
                if let Some(mappings) = self.terminology_mappings.get("advanced_to_intermediate") {
                    for (term, replacement) in mappings {
                        result = result.replace(term, replacement);
                    }
                }
            },
            _ => {
                // No simplification for advanced or expert
            },
        }
        
        Ok(result)
    }
    
    /// Expand abbreviations in the text
    fn expand_abbreviations(&self, text: &str) -> ExplanationResult<String> {
        let mut result = text.to_string();
        
        for (abbr, expansion) in &self.abbreviation_expansions {
            // Only replace standalone abbreviations (surrounded by spaces, punctuation, or string boundaries)
            let pattern = format!(r"(?i)\b{}\b", regex::escape(abbr));
            let regex = regex::Regex::new(&pattern).map_err(|e| {
                ExplanationError::TransformationError(format!("Invalid regex pattern: {}", e))
            })?;
            
            result = regex.replace_all(&result, format!("{} ({})", expansion, abbr)).to_string();
        }
        
        Ok(result)
    }
    
    /// Add examples to the text
    fn add_examples(
        &self,
        text: &str,
        parameters: &HashMap<String, String>
    ) -> ExplanationResult<String> {
        let mut result = text.to_string();
        
        if let Some(category) = parameters.get("category") {
            if let Some(examples) = self.example_templates.get(category) {
                if !examples.is_empty() {
                    let example = &examples[0]; // Just use the first example for simplicity
                    result.push_str("\n\n");
                    result.push_str(example);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Provide additional context
    fn provide_context(
        &self,
        text: &str,
        parameters: &HashMap<String, String>
    ) -> ExplanationResult<String> {
        let mut result = text.to_string();
        
        if let Some(category) = parameters.get("category") {
            if let Some(contexts) = self.context_templates.get(category) {
                if !contexts.is_empty() {
                    let context = &contexts[0]; // Just use the first context for simplicity
                    result.push_str("\n\n");
                    result.push_str(context);
                }
            }
        }
        
        Ok(result)
    }
    
    /// Reduce sentence complexity
    fn reduce_sentence_complexity(&self, text: &str) -> ExplanationResult<String> {
        // This is a simplified implementation that splits long sentences
        let sentences: Vec<&str> = text.split(|c| c == '.' || c == '!' || c == '?').collect();
        let mut result = String::new();
        
        for sentence in sentences {
            if sentence.trim().is_empty() {
                continue;
            }
            
            let words: Vec<&str> = sentence.split_whitespace().collect();
            
            if words.len() > 20 {
                // Split long sentences at logical break points
                let midpoint = words.len() / 2;
                let mut break_point = midpoint;
                
                // Look for logical break points (conjunctions)
                for i in (midpoint - 5)..(midpoint + 5) {
                    if i < words.len() {
                        let word = words[i].to_lowercase();
                        if word == "and" || word == "but" || word == "or" || word == "because" || word == "which" {
                            break_point = i;
                            break;
                        }
                    }
                }
                
                // Create two sentences
                let first_half: String = words[..break_point].join(" ");
                let second_half: String = words[break_point..].join(" ");
                
                result.push_str(&first_half);
                result.push_str(". ");
                
                // Capitalize first letter of second half
                if !second_half.is_empty() {
                    let mut chars = second_half.chars();
                    if let Some(first_char) = chars.next() {
                        let capitalized = first_char.to_uppercase().collect::<String>() + chars.as_str();
                        result.push_str(&capitalized);
                        result.push_str(". ");
                    }
                }
            } else {
                result.push_str(sentence);
                result.push_str(". ");
            }
        }
        
        Ok(result)
    }
    
    /// Add prerequisite explanations
    fn add_prerequisites(
        &self,
        text: &str,
        parameters: &HashMap<String, String>
    ) -> ExplanationResult<String> {
        let mut result = String::new();
        
        if let Some(concepts) = parameters.get("concepts") {
            let concept_list: Vec<&str> = concepts.split(',').map(|s| s.trim()).collect();
            
            for concept in concept_list {
                if let Some(explanation) = self.prerequisite_explanations.get(concept) {
                    result.push_str(explanation);
                    result.push_str("\n\n");
                }
            }
        }
        
        result.push_str(text);
        Ok(result)
    }
}

/// Semantic validator for ensuring correctness
struct SemanticValidator {
    /// Validation rules
    rules: HashMap<String, ValidationRule>,
}

/// Validation rule
struct ValidationRule {
    /// Rule identifier
    pub id: String,
    
    /// Rule description
    pub description: String,
    
    /// Rule validator function
    pub validator: Box<dyn Fn(&str) -> bool + Send + Sync>,
}

impl SemanticValidator {
    /// Create a new semantic validator
    fn new() -> Self {
        Self {
            rules: HashMap::new(),
        }
    }
    
    /// Initialize with default validation rules
    fn initialize(&mut self) {
        // Example: Ensure Big-O notation is correctly formatted
        let big_o_rule = ValidationRule {
            id: "big_o_format".to_string(),
            description: "Ensures Big-O notation is correctly formatted".to_string(),
            validator: Box::new(|text| {
                let regex = regex::Regex::new(r"O\([a-zA-Z0-9\s\^\*\+\-]+\)").unwrap();
                regex.is_match(text)
            }),
        };
        
        self.rules.insert("big_o_format".to_string(), big_o_rule);
        
        // Additional rules would be added here...
    }
    
    /// Validate text against a rule
    fn validate(&self, text: &str, rule_id: &str) -> bool {
        if let Some(rule) = self.rules.get(rule_id) {
            (rule.validator)(text)
        } else {
            true // If rule doesn't exist, consider it valid
        }
    }
    
    /// Validate text against all rules
    fn validate_all(&self, text: &str) -> HashMap<String, bool> {
        let mut results = HashMap::new();
        
        for (id, rule) in &self.rules {
            results.insert(id.clone(), (rule.validator)(text));
        }
        
        results
    }
}

impl ExplanationGeneratorImpl {
    /// Create a new explanation generator
    pub fn new(progressive_disclosure: Arc<ProgressiveDisclosure>) -> Self {
        let templates = Arc::new(RwLock::new(HashMap::new()));
        let mut template_engine = TemplateEngine::new();
        let mut linguistic_transformer = LinguisticTransformer::new();
        let mut semantic_validator = SemanticValidator::new();
        
        // Initialize components
        linguistic_transformer.initialize();
        semantic_validator.initialize();
        
        Self {
            templates,
            progressive_disclosure,
            template_engine,
            linguistic_transformer,
            semantic_validator,
        }
    }
    
    /// Load template from a string
    pub fn load_template_from_string(&self, template_str: &str) -> ExplanationResult<()> {
        let template: ExplanationTemplate = serde_json::from_str(template_str)
            .map_err(|e| ExplanationError::TemplateNotFound(format!("Invalid template JSON: {}", e)))?;
        
        if let Ok(mut templates) = self.templates.write() {
            templates.insert(template.id.clone(), template);
        }
        
        Ok(())
    }
    
    /// Find a template for a pattern
    fn find_template_for_pattern(&self, pattern_match: &PatternMatch) -> ExplanationResult<ExplanationTemplate> {
        if let Ok(templates) = self.templates.read() {
            // Find all templates for this pattern type
            let candidates: Vec<&ExplanationTemplate> = templates.values()
                .filter(|t| {
                    // Check if this template is applicable to the pattern
                    let pattern_type_match = t.conditions.iter().any(|c| {
                        if c.variable == "pattern_type" {
                            match &c.operator {
                                ConditionOperator::Equals => {
                                    if let ConditionValue::String(value) = &c.value {
                                        return value == &pattern_match.pattern_type;
                                    }
                                },
                                _ => {} // Other operators not implemented for simplicity
                            }
                        }
                        false
                    });
                    
                    pattern_type_match
                })
                .collect();
            
            if let Some(template) = candidates.first() {
                return Ok((*template).clone());
            }
        }
        
        Err(ExplanationError::TemplateNotFound(format!("No template found for pattern type: {}", pattern_match.pattern_type)))
    }
    
    /// Find a template for an anomaly
    fn find_template_for_anomaly(&self, anomaly_match: &AnomalyMatch) -> ExplanationResult<ExplanationTemplate> {
        if let Ok(templates) = self.templates.read() {
            // Find all templates for this anomaly type
            let candidates: Vec<&ExplanationTemplate> = templates.values()
                .filter(|t| {
                    // Check if this template is applicable to the anomaly
                    let anomaly_type_match = t.conditions.iter().any(|c| {
                        if c.variable == "anomaly_type" {
                            match &c.operator {
                                ConditionOperator::Equals => {
                                    if let ConditionValue::String(value) = &c.value {
                                        return value == &anomaly_match.anomaly_type;
                                    }
                                },
                                _ => {} // Other operators not implemented for simplicity
                            }
                        }
                        false
                    });
                    
                    anomaly_type_match
                })
                .collect();
            
            if let Some(template) = candidates.first() {
                return Ok((*template).clone());
            }
        }
        
        Err(ExplanationError::TemplateNotFound(format!("No template found for anomaly type: {}", anomaly_match.anomaly_type)))
    }
    
    /// Create variables from a pattern match
    fn create_variables_from_pattern(
        &self,
        pattern_match: &PatternMatch
    ) -> HashMap<String, String> {
        let mut variables = HashMap::new();
        
        variables.insert("pattern_id".to_string(), pattern_match.pattern_id.clone());
        variables.insert("pattern_type".to_string(), pattern_match.pattern_type.clone());
        variables.insert("title".to_string(), pattern_match.title.clone());
        variables.insert("summary".to_string(), pattern_match.summary.clone());
        variables.insert("confidence".to_string(), pattern_match.confidence.to_string());
        variables.insert("educational_value".to_string(), pattern_match.educational_value.to_string());
        
        // Add pattern-specific variables
        for (key, value) in &pattern_match.variables {
            variables.insert(format!("pattern_{}", key), value.clone());
        }
        
        variables
    }
    
    /// Create variables from an anomaly match
    fn create_variables_from_anomaly(
        &self,
        anomaly_match: &AnomalyMatch
    ) -> HashMap<String, String> {
        let mut variables = HashMap::new();
        
        variables.insert("anomaly_id".to_string(), anomaly_match.anomaly_id.clone());
        variables.insert("anomaly_type".to_string(), anomaly_match.anomaly_type.clone());
        variables.insert("title".to_string(), anomaly_match.title.clone());
        variables.insert("summary".to_string(), anomaly_match.summary.clone());
        variables.insert("confidence".to_string(), anomaly_match.confidence.to_string());
        variables.insert("educational_value".to_string(), anomaly_match.educational_value.to_string());
        
        // Add anomaly-specific variables
        for (key, value) in &anomaly_match.variables {
            variables.insert(format!("anomaly_{}", key), value.clone());
        }
        
        variables
    }
    
    /// Apply expertise adaptations to template
    fn apply_expertise_adaptations(
        &self,
        template: &ExplanationTemplate,
        expertise_level: &ExpertiseLevel
    ) -> ExplanationTemplate {
        let mut adapted_template = template.clone();
        
        if let Some(adaptation) = template.expertise_adaptations.get(expertise_level) {
            // Apply section inclusions/exclusions
            adapted_template.section_templates.retain(|s| {
                !adaptation.exclude_sections.contains(&s.id) &&
                (adaptation.include_sections.is_empty() || adaptation.include_sections.contains(&s.id))
            });
            
            // Apply similar filters for concepts, annotations, expressions, and examples
            // (not implemented for brevity)
        }
        
        adapted_template
    }
    
/// Apply linguistic transformations
    fn apply_linguistic_transformations(
        &self,
        text: &str,
        transformations: &[LinguisticTransformation],
        expertise_level: &ExpertiseLevel
    ) -> ExplanationResult<String> {
        let mut transformed_text = text.to_string();
        
        for transformation in transformations {
            transformed_text = self.linguistic_transformer.apply_transformation(
                &transformed_text,
                transformation,
                expertise_level
            )?;
        }
        
        Ok(transformed_text)
    }
    
    /// Generate explanation from a template
    fn generate_explanation_from_template(
        &self,
        template: &ExplanationTemplate,
        variables: &HashMap<String, String>,
        expertise_level: ExpertiseLevel
    ) -> ExplanationResult<Explanation> {
        // Apply expertise adaptations to template
        let adapted_template = self.apply_expertise_adaptations(template, &expertise_level);
        
        // Set up template engine with variables
        let mut template_engine = self.template_engine.clone();
        template_engine.set_variables(variables);
        
        // Generate title
        let title = template_engine.render(&adapted_template.title_template)?;
        
        // Generate summary
        let summary = template_engine.render(&adapted_template.summary_template)?;
        
        // Generate sections
        let mut sections = Vec::new();
        for section_template in &adapted_template.section_templates {
            // Skip sections that require higher expertise
            if section_template.min_expertise > expertise_level {
                continue;
            }
            
            // Render section title and content
            let section_title = template_engine.render(&section_template.title_template)?;
            let section_content = template_engine.render(&section_template.content_template)?;
            
            // Apply linguistic transformations based on expertise level
            let adapted_content = match expertise_level {
                ExpertiseLevel::Beginner => {
                    let transformations = vec![
                        LinguisticTransformation {
                            transformation_type: TransformationType::SimplifyTerminology,
                            parameters: HashMap::new(),
                        },
                        LinguisticTransformation {
                            transformation_type: TransformationType::ExpandAbbreviations,
                            parameters: HashMap::new(),
                        },
                        LinguisticTransformation {
                            transformation_type: TransformationType::ReduceSentenceComplexity,
                            parameters: HashMap::new(),
                        },
                        LinguisticTransformation {
                            transformation_type: TransformationType::AddExamples,
                            parameters: {
                                let mut params = HashMap::new();
                                params.insert("category".to_string(), "algorithm".to_string());
                                params
                            },
                        },
                    ];
                    
                    self.apply_linguistic_transformations(&section_content, &transformations, &expertise_level)?
                },
                ExpertiseLevel::Intermediate => {
                    let transformations = vec![
                        LinguisticTransformation {
                            transformation_type: TransformationType::SimplifyTerminology,
                            parameters: HashMap::new(),
                        },
                        LinguisticTransformation {
                            transformation_type: TransformationType::ExpandAbbreviations,
                            parameters: HashMap::new(),
                        },
                    ];
                    
                    self.apply_linguistic_transformations(&section_content, &transformations, &expertise_level)?
                },
                _ => section_content, // No transformations for advanced or expert
            };
            
            sections.push(ExplanationSection {
                title: section_title,
                content: adapted_content,
                importance: section_template.base_importance,
                order: section_template.base_order,
            });
        }
        
        // Sort sections by order
        sections.sort_by_key(|s| s.order);
        
        // Generate concept references, visual annotations, math expressions, and code examples
        // (simplified implementation for brevity)
        let concept_refs = Vec::new();
        let visual_annotations = Vec::new();
        let math_expressions = Vec::new();
        let code_examples = Vec::new();
        
        // Create explanation
        let explanation = Explanation {
            title,
            summary,
            sections,
            expertise_level,
            concept_refs,
            visual_annotations,
            math_expressions,
            code_examples,
            metadata: HashMap::new(),
        };
        
        // Validate explanation semantics
        self.validate_explanation(&explanation)?;
        
        Ok(explanation)
    }
    
    /// Validate explanation semantics
    fn validate_explanation(&self, explanation: &Explanation) -> ExplanationResult<()> {
        // Validate each section content
        for section in &explanation.sections {
            let results = self.semantic_validator.validate_all(&section.content);
            
            // Check for validation failures
            for (rule_id, valid) in results {
                if !valid {
                    return Err(ExplanationError::ValidationError(
                        format!("Validation failed for rule {}: {:?}", rule_id, section.title)
                    ));
                }
            }
        }
        
        Ok(())
    }
}

impl ExplanationGenerator for ExplanationGeneratorImpl {
    fn generate_explanation(
        &self,
        pattern_match: &PatternMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation {
        // Find template for this pattern
        let template = match self.find_template_for_pattern(pattern_match) {
            Ok(template) => template,
            Err(e) => {
                warn!("Failed to find template: {}", e);
                // Fall back to a default explanation
                return self.generate_fallback_explanation(pattern_match, expertise_level);
            }
        };
        
        // Create variables from pattern match
        let variables = self.create_variables_from_pattern(pattern_match);
        
        // Generate explanation from template
        match self.generate_explanation_from_template(&template, &variables, expertise_level) {
            Ok(explanation) => explanation,
            Err(e) => {
                warn!("Failed to generate explanation: {}", e);
                // Fall back to a default explanation
                self.generate_fallback_explanation(pattern_match, expertise_level)
            }
        }
    }
    
    fn generate_explanation_for_anomaly(
        &self,
        anomaly_match: &AnomalyMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation {
        // Find template for this anomaly
        let template = match self.find_template_for_anomaly(anomaly_match) {
            Ok(template) => template,
            Err(e) => {
                warn!("Failed to find template: {}", e);
                // Fall back to a default explanation
                return self.generate_fallback_explanation_for_anomaly(anomaly_match, expertise_level);
            }
        };
        
        // Create variables from anomaly match
        let variables = self.create_variables_from_anomaly(anomaly_match);
        
        // Generate explanation from template
        match self.generate_explanation_from_template(&template, &variables, expertise_level) {
            Ok(explanation) => explanation,
            Err(e) => {
                warn!("Failed to generate explanation: {}", e);
                // Fall back to a default explanation
                self.generate_fallback_explanation_for_anomaly(anomaly_match, expertise_level)
            }
        }
    }
    
    fn generate_explanation_for_state(
        &self,
        state: &AlgorithmState,
        algorithm: &dyn Algorithm,
        expertise_level: ExpertiseLevel
    ) -> Explanation {
        // Generate a basic explanation for the algorithm state
        // This is a simplified implementation
        let title = format!("State of {} at step {}", algorithm.name(), state.step);
        let summary = format!("Current state of the {} algorithm at execution step {}.", 
                             algorithm.name(), state.step);
        
        let mut sections = Vec::new();
        
        // Add basic state information section
        sections.push(ExplanationSection {
            title: "Current State".to_string(),
            content: format!("The algorithm is currently at step {}. The current node being \
                            processed is {:?}.", state.step, state.current_node),
            importance: 1.0,
            order: 0,
        });
        
        // Add open set section if applicable
        if !state.open_set.is_empty() {
            let open_set_content = match expertise_level {
                ExpertiseLevel::Beginner => {
                    format!("The algorithm is considering {} nodes for future exploration.", 
                           state.open_set.len())
                },
                ExpertiseLevel::Intermediate => {
                    format!("The open set contains {} nodes that are queued for future exploration: {:?}.", 
                           state.open_set.len(), state.open_set)
                },
                _ => {
                    format!("The open set contains {} nodes queued for exploration. The frontier \
                            of the search space is defined by these nodes: {:?}.", 
                           state.open_set.len(), state.open_set)
                }
            };
            
            sections.push(ExplanationSection {
                title: "Open Set".to_string(),
                content: open_set_content,
                importance: 0.8,
                order: 1,
            });
        }
        
        // Add closed set section if applicable
        if !state.closed_set.is_empty() {
            let closed_set_content = match expertise_level {
                ExpertiseLevel::Beginner => {
                    format!("The algorithm has already explored {} nodes.", 
                           state.closed_set.len())
                },
                ExpertiseLevel::Intermediate => {
                    format!("The closed set contains {} nodes that have been fully explored: {:?}.", 
                           state.closed_set.len(), state.closed_set)
                },
                _ => {
                    format!("The closed set contains {} nodes that have been fully explored. These \
                            nodes have had all their neighbors examined: {:?}.", 
                           state.closed_set.len(), state.closed_set)
                }
            };
            
            sections.push(ExplanationSection {
                title: "Closed Set".to_string(),
                content: closed_set_content,
                importance: 0.7,
                order: 2,
            });
        }
        
        // Add custom data section if applicable
        if !state.data.is_empty() {
            let mut data_lines = Vec::new();
            for (key, value) in &state.data {
                data_lines.push(format!("- {}: {}", key, value));
            }
            
            let data_content = match expertise_level {
                ExpertiseLevel::Beginner => {
                    format!("The algorithm is tracking additional information:")
                },
                _ => {
                    format!("The algorithm is tracking the following additional data:\n\n{}", 
                           data_lines.join("\n"))
                }
            };
            
            sections.push(ExplanationSection {
                title: "Algorithm Data".to_string(),
                content: data_content,
                importance: 0.6,
                order: 3,
            });
        }
        
        Explanation {
            title,
            summary,
            sections,
            expertise_level,
            concept_refs: Vec::new(),
            visual_annotations: Vec::new(),
            math_expressions: Vec::new(),
            code_examples: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    fn adapt_explanation(
        &self,
        explanation: &Explanation,
        target_expertise: ExpertiseLevel
    ) -> ExplanationResult<Explanation> {
        // If already at target expertise, return a clone
        if explanation.expertise_level == target_expertise {
            return Ok(explanation.clone());
        }
        
        let mut adapted = explanation.clone();
        adapted.expertise_level = target_expertise.clone();
        
        // Adapt sections based on expertise
        if target_expertise < explanation.expertise_level {
            // Going to lower expertise: simplify and possibly remove sections
            let mut simplified_sections = Vec::new();
            
            for section in &explanation.sections {
                // Skip highly technical sections when adapting to beginner
                if target_expertise == ExpertiseLevel::Beginner && section.importance < 0.5 {
                    continue;
                }
                
                // Apply linguistic transformations based on target expertise
                let transformations = match target_expertise {
                    ExpertiseLevel::Beginner => {
                        vec![
                            LinguisticTransformation {
                                transformation_type: TransformationType::SimplifyTerminology,
                                parameters: HashMap::new(),
                            },
                            LinguisticTransformation {
                                transformation_type: TransformationType::ExpandAbbreviations,
                                parameters: HashMap::new(),
                            },
                            LinguisticTransformation {
                                transformation_type: TransformationType::ReduceSentenceComplexity,
                                parameters: HashMap::new(),
                            },
                            LinguisticTransformation {
                                transformation_type: TransformationType::AddExamples,
                                parameters: {
                                    let mut params = HashMap::new();
                                    params.insert("category".to_string(), "algorithm".to_string());
                                    params
                                },
                            },
                        ]
                    },
                    ExpertiseLevel::Intermediate => {
                        vec![
                            LinguisticTransformation {
                                transformation_type: TransformationType::SimplifyTerminology,
                                parameters: HashMap::new(),
                            },
                            LinguisticTransformation {
                                transformation_type: TransformationType::ExpandAbbreviations,
                                parameters: HashMap::new(),
                            },
                        ]
                    },
                    _ => Vec::new(), // No transformations for advanced or expert
                };
                
                // Apply transformations
                match self.apply_linguistic_transformations(&section.content, &transformations, &target_expertise) {
                    Ok(simplified_content) => {
                        let mut simplified_section = section.clone();
                        simplified_section.content = simplified_content;
                        simplified_sections.push(simplified_section);
                    },
                    Err(e) => {
                        warn!("Failed to simplify section: {}", e);
                        simplified_sections.push(section.clone());
                    }
                }
            }
            
            adapted.sections = simplified_sections;
        } else {
            // Going to higher expertise: not implemented in this example
            // Would require additional domain knowledge to add technical details
        }
        
        // Validate the adapted explanation
        self.validate_explanation(&adapted)?;
        
        Ok(adapted)
    }
    
    fn combine_explanations(
        &self,
        explanations: &[Explanation]
    ) -> ExplanationResult<Explanation> {
        if explanations.is_empty() {
            return Err(ExplanationError::ValidationError("Cannot combine empty explanations".to_string()));
        }
        
        if explanations.len() == 1 {
            return Ok(explanations[0].clone());
        }
        
        // Use the first explanation as a base
        let base = &explanations[0];
        let mut combined = base.clone();
        
        // Update title and summary
        combined.title = format!("Combined: {}", base.title);
        combined.summary = format!("This explanation combines insights from multiple sources.");
        
        // Combine sections from other explanations
        for explanation in explanations.iter().skip(1) {
            for section in &explanation.sections {
                // Check if this section already exists by title
                if let Some(existing) = combined.sections.iter_mut().find(|s| s.title == section.title) {
                    // Append content to existing section
                    existing.content.push_str("\n\n");
                    existing.content.push_str(&section.content);
                    
                    // Update importance and order if higher
                    if section.importance > existing.importance {
                        existing.importance = section.importance;
                    }
                    if section.order < existing.order {
                        existing.order = section.order;
                    }
                } else {
                    // Add as a new section
                    combined.sections.push(section.clone());
                }
            }
            
            // Combine concept references
            for concept in &explanation.concept_refs {
                if !combined.concept_refs.iter().any(|c| c.name == concept.name) {
                    combined.concept_refs.push(concept.clone());
                }
            }
            
            // Combine visual annotations
            for annotation in &explanation.visual_annotations {
                combined.visual_annotations.push(annotation.clone());
            }
            
            // Combine math expressions
            for expression in &explanation.math_expressions {
                if !combined.math_expressions.iter().any(|e| e.id == expression.id) {
                    combined.math_expressions.push(expression.clone());
                }
            }
            
            // Combine code examples
            for example in &explanation.code_examples {
                if !combined.code_examples.iter().any(|e| e.id == example.id) {
                    combined.code_examples.push(example.clone());
                }
            }
        }
        
        // Sort sections by order
        combined.sections.sort_by_key(|s| s.order);
        
        // Validate the combined explanation
        self.validate_explanation(&combined)?;
        
        Ok(combined)
    }
    
    /// Generate a fallback explanation for pattern match
    fn generate_fallback_explanation(
        &self,
        pattern_match: &PatternMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation {
        // Simple fallback implementation when no template is found
        Explanation {
            title: pattern_match.title.clone(),
            summary: pattern_match.summary.clone(),
            sections: vec![
                ExplanationSection {
                    title: "Pattern Description".to_string(),
                    content: format!("This is a pattern of type {}.", pattern_match.pattern_type),
                    importance: 1.0,
                    order: 0,
                }
            ],
            expertise_level,
            concept_refs: Vec::new(),
            visual_annotations: Vec::new(),
            math_expressions: Vec::new(),
            code_examples: Vec::new(),
            metadata: HashMap::new(),
        }
    }
    
    /// Generate a fallback explanation for anomaly match
    fn generate_fallback_explanation_for_anomaly(
        &self,
        anomaly_match: &AnomalyMatch,
        expertise_level: ExpertiseLevel
    ) -> Explanation {
        // Simple fallback implementation when no template is found
        Explanation {
            title: anomaly_match.title.clone(),
            summary: anomaly_match.summary.clone(),
            sections: vec![
                ExplanationSection {
                    title: "Anomaly Description".to_string(),
                    content: format!("This is an anomaly of type {}.", anomaly_match.anomaly_type),
                    importance: 1.0,
                    order: 0,
                }
            ],
            expertise_level,
            concept_refs: Vec::new(),
            visual_annotations: Vec::new(),
            math_expressions: Vec::new(),
            code_examples: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Unit tests for the explanation generation system
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::education::progressive::ExpertiseLevel;
    
    /// Test template rendering
    #[test]
    fn test_template_rendering() {
        let mut engine = TemplateEngine::new();
        engine.set_variable("name", "World");
        
        let template = "Hello, {name}!";
        let result = engine.render(template).unwrap();
        
        assert_eq!(result, "Hello, World!");
    }
    
    /// Test conditional rendering
    #[test]
    fn test_conditional_rendering() {
        let mut engine = TemplateEngine::new();
        engine.set_variable("show_section", "true");
        engine.set_variable("hide_section", "false");
        
        let template = "Start {if:show_section}Visible{endif} {ifnot:hide_section}Also Visible{endif} End";
        let result = engine.render(template).unwrap();
        
        assert_eq!(result, "Start Visible Also Visible End");
    }
    
    /// Test loop rendering
    #[test]
    fn test_loop_rendering() {
        let mut engine = TemplateEngine::new();
        engine.set_variable("items", "apple,banana,cherry");
        
        let template = "Fruits: {foreach:item in items}- {item}\n{endforeach}";
        let result = engine.render(template).unwrap();
        
        assert_eq!(result, "Fruits: - apple\n- banana\n- cherry\n");
    }
    
    /// Test linguistic transformations
    #[test]
    fn test_linguistic_transformations() {
        // TODO: Add comprehensive tests for linguistic transformations
    }
    
    /// Test explanation generation
    #[test]
    fn test_explanation_generation() {
        // TODO: Add comprehensive tests for explanation generation
    }
    
    /// Test expertise adaptation
    #[test]
    fn test_expertise_adaptation() {
        // TODO: Add comprehensive tests for expertise adaptation
    }
    
    /// Test explanation combination
    #[test]
    fn test_explanation_combination() {
        // TODO: Add comprehensive tests for explanation combination
    }
}
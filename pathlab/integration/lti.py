"""
Chronos LTI Provider: Standards-compliant LTI 1.3 implementation with algorithm-specific extensions

This module provides a comprehensive Learning Tools Interoperability (LTI) 1.3 integration
for the Chronos Algorithmic Observatory, enabling secure and semantically rich integration
with educational systems while preserving algorithm-specific educational semantics.

Copyright (c) 2025 Mohammad Atashi <mohammadaliatashi@icloud.com>
"""

import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union, Callable

import jwt
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pylti1p3.contrib.django import DjangoOIDCLogin, DjangoMessageLaunch
from pylti1p3.deep_linking import DeepLinking
from pylti1p3.grade import Grade
from pylti1p3.lineitem import LineItem
from pylti1p3.registration import Registration
from pylti1p3.resource_link import ResourceLink
from pylti1p3.tool_config import ToolConfDict
from pylti1p3.tool_config.claim import ClaimValidator

from pathlab.education.learning_path import LearningPath
from pathlab.education.assessment import AssessmentItem, AssessmentResult


logger = logging.getLogger(__name__)


class AlgorithmLTIExtension:
    """
    Extension to LTI standard for algorithm-specific educational semantics.
    
    This class implements the Chronos-specific semantic extensions to the
    LTI standard, enabling rich educational data exchange while maintaining
    standards compliance.
    """
    
    # Algorithm semantic namespaces for LTI claims
    CHRONOS_NAMESPACE = "https://chronos.algorithmicobservatory.org/lti/extensions"
    ALGORITHM_NAMESPACE = f"{CHRONOS_NAMESPACE}/algorithm"
    TEMPORAL_NAMESPACE = f"{CHRONOS_NAMESPACE}/temporal"
    VISUALIZATION_NAMESPACE = f"{CHRONOS_NAMESPACE}/visualization"
    
    def __init__(self):
        """Initialize the algorithm LTI extension."""
        self._validators = {
            f"{self.ALGORITHM_NAMESPACE}/algorithm_type": self._validate_algorithm_type,
            f"{self.ALGORITHM_NAMESPACE}/algorithm_params": self._validate_algorithm_params,
            f"{self.TEMPORAL_NAMESPACE}/timeline_length": self._validate_timeline_length,
            f"{self.TEMPORAL_NAMESPACE}/branch_count": self._validate_branch_count,
            f"{self.VISUALIZATION_NAMESPACE}/perspective": self._validate_perspective,
        }
    
    def register_validators(self, claim_validator: ClaimValidator) -> None:
        """
        Register custom claim validators with the LTI system.
        
        Args:
            claim_validator: The LTI claim validator to register with
        """
        for claim, validator in self._validators.items():
            claim_validator.register_validator(claim, validator)
    
    def _validate_algorithm_type(self, claim_value: str) -> bool:
        """
        Validate the algorithm type claim.
        
        Args:
            claim_value: The algorithm type string
            
        Returns:
            True if valid, False otherwise
        """
        valid_types = ["path_finding", "graph", "sorting", "dynamic_programming"]
        return claim_value in valid_types
    
    def _validate_algorithm_params(self, claim_value: Dict[str, Any]) -> bool:
        """
        Validate the algorithm parameters claim.
        
        Args:
            claim_value: Dictionary of algorithm parameters
            
        Returns:
            True if valid, False otherwise
        """
        if not isinstance(claim_value, dict):
            return False
        
        # Validate structure based on algorithm type
        if "type" not in claim_value:
            return False
            
        algorithm_type = claim_value["type"]
        
        if algorithm_type == "path_finding":
            return "heuristic" in claim_value and "weight_function" in claim_value
        elif algorithm_type == "sorting":
            return "comparison_function" in claim_value
        elif algorithm_type == "graph":
            return "operation" in claim_value
        
        return True
    
    def _validate_timeline_length(self, claim_value: int) -> bool:
        """
        Validate the timeline length claim.
        
        Args:
            claim_value: The timeline length value
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(claim_value, int) and 0 <= claim_value <= 100000
    
    def _validate_branch_count(self, claim_value: int) -> bool:
        """
        Validate the branch count claim.
        
        Args:
            claim_value: The branch count value
            
        Returns:
            True if valid, False otherwise
        """
        return isinstance(claim_value, int) and 0 <= claim_value <= 100
    
    def _validate_perspective(self, claim_value: str) -> bool:
        """
        Validate the visualization perspective claim.
        
        Args:
            claim_value: The perspective string
            
        Returns:
            True if valid, False otherwise
        """
        valid_perspectives = ["decision", "heuristic", "progress", "state_space"]
        return claim_value in valid_perspectives
    
    def enhance_launch_data(self, launch_data: Dict[str, Any], 
                           algorithm_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhance LTI launch data with algorithm-specific information.
        
        Args:
            launch_data: The base LTI launch data
            algorithm_data: The algorithm-specific data to add
            
        Returns:
            Enhanced launch data dictionary
        """
        enhanced_data = launch_data.copy()
        
        # Add algorithm type if present
        if "algorithm_type" in algorithm_data:
            enhanced_data[f"{self.ALGORITHM_NAMESPACE}/algorithm_type"] = algorithm_data["algorithm_type"]
        
        # Add algorithm parameters if present
        if "algorithm_params" in algorithm_data:
            enhanced_data[f"{self.ALGORITHM_NAMESPACE}/algorithm_params"] = algorithm_data["algorithm_params"]
        
        # Add temporal information if present
        if "timeline_length" in algorithm_data:
            enhanced_data[f"{self.TEMPORAL_NAMESPACE}/timeline_length"] = algorithm_data["timeline_length"]
        
        if "branch_count" in algorithm_data:
            enhanced_data[f"{self.TEMPORAL_NAMESPACE}/branch_count"] = algorithm_data["branch_count"]
        
        # Add visualization information if present
        if "perspective" in algorithm_data:
            enhanced_data[f"{self.VISUALIZATION_NAMESPACE}/perspective"] = algorithm_data["perspective"]
        
        return enhanced_data
    
    def extract_algorithm_data(self, launch_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract algorithm-specific data from LTI launch data.
        
        Args:
            launch_data: The LTI launch data
            
        Returns:
            Dictionary of algorithm-specific data
        """
        algorithm_data = {}
        
        # Extract algorithm type
        algorithm_type_key = f"{self.ALGORITHM_NAMESPACE}/algorithm_type"
        if algorithm_type_key in launch_data:
            algorithm_data["algorithm_type"] = launch_data[algorithm_type_key]
        
        # Extract algorithm parameters
        algorithm_params_key = f"{self.ALGORITHM_NAMESPACE}/algorithm_params"
        if algorithm_params_key in launch_data:
            algorithm_data["algorithm_params"] = launch_data[algorithm_params_key]
        
        # Extract temporal information
        timeline_length_key = f"{self.TEMPORAL_NAMESPACE}/timeline_length"
        if timeline_length_key in launch_data:
            algorithm_data["timeline_length"] = launch_data[timeline_length_key]
        
        branch_count_key = f"{self.TEMPORAL_NAMESPACE}/branch_count"
        if branch_count_key in launch_data:
            algorithm_data["branch_count"] = launch_data[branch_count_key]
        
        # Extract visualization information
        perspective_key = f"{self.VISUALIZATION_NAMESPACE}/perspective"
        if perspective_key in launch_data:
            algorithm_data["perspective"] = launch_data[perspective_key]
        
        return algorithm_data


class ChronosLTIProvider:
    """
    LTI 1.3 provider implementation for Chronos Algorithmic Observatory.
    
    This class provides a standards-compliant LTI 1.3 provider with extensions
    for algorithm-specific educational semantics, enabling secure integration
    with Learning Management Systems and other educational tools.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the LTI provider.
        
        Args:
            config: Configuration dictionary for the LTI provider
        """
        self.config = config
        self.tool_config = ToolConfDict(config)
        self.extension = AlgorithmLTIExtension()
        
        # Register extension validators
        if hasattr(self.tool_config, 'claim_validator'):
            self.extension.register_validators(self.tool_config.claim_validator)
    
    def handle_login_request(self, request) -> Tuple[str, Dict[str, Any]]:
        """
        Handle LTI login request.
        
        Args:
            request: The HTTP request object
            
        Returns:
            Tuple of (redirect_url, state)
        """
        oidc_login = DjangoOIDCLogin(request, self.tool_config)
        return oidc_login.enable_check_cookies().redirect()
    
    def handle_launch_request(self, request) -> 'MessageLaunchResult':
        """
        Handle LTI launch request.
        
        Args:
            request: The HTTP request object
            
        Returns:
            MessageLaunchResult instance
        """
        message_launch = DjangoMessageLaunch(request, self.tool_config)
        launch_data = message_launch.get_launch_data()
        
        # Extract algorithm-specific data
        algorithm_data = self.extension.extract_algorithm_data(launch_data)
        
        return MessageLaunchResult(message_launch, algorithm_data)
    
    def create_deep_linking_response(self, 
                                    launch_result: 'MessageLaunchResult',
                                    content_items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create deep linking response for content selection.
        
        Args:
            launch_result: The message launch result
            content_items: List of content items to include
            
        Returns:
            Deep linking response data
        """
        deep_linking = DeepLinking(launch_result.message_launch)
        return deep_linking.get_response_jwt(content_items)
    
    def create_algorithm_content_item(self, 
                                     algorithm_type: str,
                                     title: str,
                                     description: str,
                                     algorithm_params: Dict[str, Any] = None,
                                     custom_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create a content item for an algorithm.
        
        Args:
            algorithm_type: Type of algorithm
            title: Title of the content item
            description: Description of the content item
            algorithm_params: Algorithm parameters
            custom_params: Custom parameters
            
        Returns:
            Content item dictionary
        """
        content_item = {
            "type": "ltiResourceLink",
            "title": title,
            "url": self.config.get("launch_url", ""),
            "presentation": {
                "documentTarget": "iframe",
            },
            "custom": custom_params or {},
        }
        
        # Add algorithm-specific data
        algorithm_data = {
            "algorithm_type": algorithm_type
        }
        
        if algorithm_params:
            algorithm_data["algorithm_params"] = algorithm_params
            
        # Enhance custom parameters with algorithm data
        content_item["custom"].update({
            f"{AlgorithmLTIExtension.ALGORITHM_NAMESPACE}/algorithm_type": algorithm_type
        })
        
        if algorithm_params:
            content_item["custom"][f"{AlgorithmLTIExtension.ALGORITHM_NAMESPACE}/algorithm_params"] = json.dumps(algorithm_params)
        
        return content_item
    
    def send_outcome(self, 
                    launch_result: 'MessageLaunchResult',
                    score: float,
                    assessment_result: AssessmentResult) -> bool:
        """
        Send outcome data back to the Tool Consumer.
        
        Args:
            launch_result: The message launch result
            score: The score value (0.0 to 1.0)
            assessment_result: The assessment result data
            
        Returns:
            True if successful, False otherwise
        """
        if not 0.0 <= score <= 1.0:
            raise ValueError("Score must be between 0.0 and 1.0")
        
        # Create grade object
        grade = Grade()
        grade.set_score_given(score)
        grade.set_score_maximum(1.0)
        grade.set_timestamp(datetime.utcnow().isoformat())
        grade.set_activity_progress("Completed")
        grade.set_grading_progress("FullyGraded")
        
        # Add Chronos-specific data
        grade.set_comment(json.dumps({
            "assessment_type": assessment_result.assessment_type,
            "mastery_level": assessment_result.mastery_level,
            "completion_time": assessment_result.completion_time,
            "algorithm_type": assessment_result.algorithm_type,
            "skill_areas": assessment_result.skill_areas,
            "recommendations": assessment_result.recommendations
        }))
        
        # Send grade
        result = launch_result.message_launch.set_auto_create_resource_link(True) \
                              .get_ags() \
                              .put_grade(grade)
        
        return result.is_success()
    
    def create_line_item(self, 
                        launch_result: 'MessageLaunchResult',
                        title: str,
                        score_maximum: float = 1.0,
                        resource_id: str = None,
                        tag: str = None) -> Dict[str, Any]:
        """
        Create a line item for grading.
        
        Args:
            launch_result: The message launch result
            title: Title of the line item
            score_maximum: Maximum score for the line item
            resource_id: Resource ID for the line item
            tag: Tag for the line item
            
        Returns:
            Line item creation result
        """
        line_item = LineItem()
        line_item.set_label(title)
        line_item.set_score_maximum(score_maximum)
        
        if resource_id:
            line_item.set_resource_id(resource_id)
            
        if tag:
            line_item.set_tag(tag)
        
        result = launch_result.message_launch.get_ags().put_grade(line_item)
        return result.get_data()


class MessageLaunchResult:
    """
    Result of an LTI message launch.
    
    This class encapsulates the result of an LTI message launch, providing
    convenient access to launch data and services.
    """
    
    def __init__(self, message_launch, algorithm_data: Dict[str, Any]):
        """
        Initialize the message launch result.
        
        Args:
            message_launch: The message launch object
            algorithm_data: Algorithm-specific data extracted from the launch
        """
        self.message_launch = message_launch
        self.algorithm_data = algorithm_data
        self.launch_data = message_launch.get_launch_data()
        
    def get_user_id(self) -> str:
        """
        Get the user ID from the launch.
        
        Returns:
            User ID string
        """
        return self.launch_data.get("sub", "")
    
    def get_roles(self) -> List[str]:
        """
        Get the user roles from the launch.
        
        Returns:
            List of role strings
        """
        return self.launch_data.get("https://purl.imsglobal.org/spec/lti/claim/roles", [])
    
    def is_instructor(self) -> bool:
        """
        Check if the user is an instructor.
        
        Returns:
            True if the user is an instructor, False otherwise
        """
        instructor_roles = [
            "http://purl.imsglobal.org/vocab/lis/v2/membership#Instructor",
            "http://purl.imsglobal.org/vocab/lis/v2/institution/person#Instructor",
            "http://purl.imsglobal.org/vocab/lis/v2/membership#ContentDeveloper",
            "http://purl.imsglobal.org/vocab/lis/v2/membership#TeachingAssistant"
        ]
        
        roles = self.get_roles()
        return any(role in instructor_roles for role in roles)
    
    def is_student(self) -> bool:
        """
        Check if the user is a student.
        
        Returns:
            True if the user is a student, False otherwise
        """
        student_roles = [
            "http://purl.imsglobal.org/vocab/lis/v2/membership#Learner",
            "http://purl.imsglobal.org/vocab/lis/v2/institution/person#Student"
        ]
        
        roles = self.get_roles()
        return any(role in student_roles for role in roles)
    
    def get_context_id(self) -> str:
        """
        Get the context ID from the launch.
        
        Returns:
            Context ID string
        """
        return self.launch_data.get("https://purl.imsglobal.org/spec/lti/claim/context", {}).get("id", "")
    
    def get_resource_link_id(self) -> str:
        """
        Get the resource link ID from the launch.
        
        Returns:
            Resource link ID string
        """
        return self.launch_data.get("https://purl.imsglobal.org/spec/lti/claim/resource_link", {}).get("id", "")
    
    def get_custom_parameters(self) -> Dict[str, Any]:
        """
        Get custom parameters from the launch.
        
        Returns:
            Dictionary of custom parameters
        """
        return self.launch_data.get("https://purl.imsglobal.org/spec/lti/claim/custom", {})
    
    def get_algorithm_type(self) -> Optional[str]:
        """
        Get the algorithm type from the launch.
        
        Returns:
            Algorithm type string or None if not present
        """
        return self.algorithm_data.get("algorithm_type")
    
    def get_algorithm_params(self) -> Optional[Dict[str, Any]]:
        """
        Get the algorithm parameters from the launch.
        
        Returns:
            Dictionary of algorithm parameters or None if not present
        """
        return self.algorithm_data.get("algorithm_params")
    
    def get_timeline_length(self) -> Optional[int]:
        """
        Get the timeline length from the launch.
        
        Returns:
            Timeline length or None if not present
        """
        return self.algorithm_data.get("timeline_length")
    
    def get_branch_count(self) -> Optional[int]:
        """
        Get the branch count from the launch.
        
        Returns:
            Branch count or None if not present
        """
        return self.algorithm_data.get("branch_count")
    
    def get_perspective(self) -> Optional[str]:
        """
        Get the visualization perspective from the launch.
        
        Returns:
            Perspective string or None if not present
        """
        return self.algorithm_data.get("perspective")


# Configuration utilities
class LTIConfigGenerator:
    """
    Utility for generating LTI configuration.
    
    This class provides utilities for generating LTI configuration,
    including key generation and configuration formatting.
    """
    
    @staticmethod
    def generate_key_pair() -> Tuple[str, str]:
        """
        Generate RSA key pair for LTI use.
        
        Returns:
            Tuple of (private_key_pem, public_key_pem)
        """
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption()
        ).decode('utf-8')
        
        public_key = private_key.public_key()
        public_pem = public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ).decode('utf-8')
        
        return private_pem, public_pem
    
    @staticmethod
    def create_tool_configuration(client_id: str, 
                                 private_key: str,
                                 public_key: str,
                                 tool_name: str = "Chronos Algorithmic Observatory",
                                 deployment_ids: List[str] = None,
                                 launch_url: str = "",
                                 redirect_uris: List[str] = None,
                                 jwks_url: str = None) -> Dict[str, Any]:
        """
        Create tool configuration dictionary.
        
        Args:
            client_id: The LTI client ID
            private_key: The private key PEM string
            public_key: The public key PEM string
            tool_name: The name of the tool
            deployment_ids: List of deployment IDs
            launch_url: URL for LTI launches
            redirect_uris: List of redirect URIs
            jwks_url: URL for JWKS endpoint
            
        Returns:
            Tool configuration dictionary
        """
        config = {
            "iss": client_id,
            "client_id": client_id,
            "auth_login_url": launch_url.replace("launch", "login") if launch_url else "",
            "auth_token_url": launch_url.replace("launch", "token") if launch_url else "",
            "key_set_url": jwks_url or "",
            "private_key_file": "private.key",  # This will be overridden
            "private_key": private_key,
            "public_key": public_key,
            "deployment_ids": deployment_ids or [],
            "tool_name": tool_name,
            "launch_url": launch_url,
            "redirect_uris": redirect_uris or [],
        }
        
        return config
    
    @staticmethod
    def generate_deployment_id() -> str:
        """
        Generate a deployment ID.
        
        Returns:
            Deployment ID string
        """
        return str(uuid.uuid4())
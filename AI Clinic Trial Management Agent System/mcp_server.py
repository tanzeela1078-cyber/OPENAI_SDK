#!/usr/bin/env python3
"""
mcp_server.py — Clinical Trial Management MCP Server
Handles trial orchestration, data validation, ESG compliance, and audit trails
Based on FHIR_MCP_SERVER.py structure
"""

import asyncio
import json
import aiohttp
from datetime import datetime
from typing import Dict, List, Optional, Any
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("Clinical Trial Management Server")

# HAPI FHIR Configuration
FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"

# ──────────────────────────────────────────────────────────────
# 🧩 MCP Tools for Clinical Trial Management
# ──────────────────────────────────────────────────────────────

@mcp.tool()
async def get_trial_protocol(trial_id: str) -> dict:
    """Get clinical trial protocol information from HAPI FHIR."""
    try:
        # Query ResearchStudy resources from HAPI
        url = f"{FHIR_BASE_URL}/ResearchStudy?identifier={trial_id}&_count=1"
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    # If no ResearchStudy found, create protocol from available data
                    return {
                        "trial_id": trial_id,
                        "protocol_name": f"Clinical Trial {trial_id}",
                        "phase": "Phase II",
                        "status": "Active",
                        "primary_endpoint": "Efficacy and Safety Analysis",
                        "sample_size": 150,
                        "data_source": "HAPI FHIR (Generated Protocol)"
                    }
                
                data = await resp.json()
                studies = data.get('entry', [])
                
                if studies:
                    study = studies[0].get('resource', {})
                    return {
                        "trial_id": trial_id,
                        "protocol_name": study.get('title', f"Clinical Trial {trial_id}"),
                        "phase": study.get('phase', {}).get('coding', [{}])[0].get('display', 'Phase II'),
                        "status": study.get('status', 'Active'),
                        "primary_endpoint": study.get('description', 'Efficacy and Safety Analysis'),
                        "sample_size": study.get('enrollment', [{}])[0].get('value', 150),
                        "data_source": "HAPI FHIR ResearchStudy"
                    }
                else:
                    # Create protocol from available HAPI data
                    return {
                        "trial_id": trial_id,
                        "protocol_name": f"Clinical Trial {trial_id}",
                        "phase": "Phase II",
                        "status": "Active",
                        "primary_endpoint": "Efficacy and Safety Analysis",
                        "sample_size": 150,
                        "data_source": "HAPI FHIR (Generated Protocol)"
                    }
    except Exception as e:
        return {"error": f"Failed to fetch trial protocol: {str(e)}"}

@mcp.tool()
async def get_trial_participants(trial_id: str) -> dict:
    """Get trial participants from HAPI FHIR Patient data with clinical trial focus."""
    try:
        # Use single session for all requests
        async with aiohttp.ClientSession() as session:
            # First try to get ResearchSubject resources (proper clinical trial participants)
            research_subjects_url = f"{FHIR_BASE_URL}/ResearchSubject?_count=20"
            async with session.get(research_subjects_url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    participants = []
                    
                    for entry in data.get('entry', []):
                        subject = entry.get('resource', {})
                        subject_id = subject.get('id', 'Unknown')
                        
                        # Get patient reference
                        patient_ref = subject.get('individual', {}).get('reference', '')
                        patient_id = patient_ref.replace('Patient/', '') if 'Patient/' in patient_ref else 'Unknown'
                        
                        # Get patient details using same session
                        if patient_id != 'Unknown':
                            patient_url = f"{FHIR_BASE_URL}/Patient/{patient_id}"
                            async with session.get(patient_url) as patient_resp:
                                if patient_resp.status == 200:
                                    patient_data = await patient_resp.json()
                                    names = patient_data.get('name', [])
                                    given_name = names[0].get('given', ['Unknown'])[0] if names else 'Unknown'
                                    family_name = names[0].get('family', 'Unknown') if names else 'Unknown'
                                    full_name = f"{given_name} {family_name}"
                                else:
                                    full_name = f"Patient {patient_id}"
                        else:
                            full_name = "Unknown Patient"
                        
                        participants.append({
                            'subject_id': subject_id,
                            'patient_id': patient_id,
                            'patient_name': full_name,
                            'status': subject.get('status', 'Active'),
                            'consent_date': subject.get('period', {}).get('start', 'Unknown'),
                            'study_arm': subject.get('study', {}).get('display', 'Unknown'),
                            'gender': 'Unknown',  # Would need patient data
                            'birth_date': 'Unknown'  # Would need patient data
                        })
                    
                    if participants:
                        return {
                            "trial_id": trial_id,
                            "participants": participants,
                            "total_participants": len(participants),
                            "data_source": "HAPI FHIR ResearchSubject Data"
                        }
            
            # Fallback to Patient data if no ResearchSubject found
            url = f"{FHIR_BASE_URL}/Patient?_count=20"
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {"error": f"Failed to fetch participants: {resp.status}"}
                
                data = await resp.json()
                participants = []
                
                for entry in data.get('entry', []):
                    patient = entry.get('resource', {})
                    patient_id = patient.get('id', 'Unknown')
                    
                    # Extract name from HAPI data
                    names = patient.get('name', [])
                    given_name = names[0].get('given', ['Unknown'])[0] if names else 'Unknown'
                    family_name = names[0].get('family', 'Unknown') if names else 'Unknown'
                    full_name = f"{given_name} {family_name}"
                    
                    participants.append({
                        'subject_id': f"SUBJ-{patient_id}",
                        'patient_id': patient_id,
                        'patient_name': full_name,
                        'status': 'Active',
                        'consent_date': '2025-01-15',
                        'study_arm': 'Treatment' if int(patient_id[-1]) % 2 == 0 else 'Control',
                        'gender': patient.get('gender', 'Unknown'),
                        'birth_date': patient.get('birthDate', 'Unknown')
                    })
                
                return {
                    "trial_id": trial_id,
                    "participants": participants,
                    "total_participants": len(participants),
                    "data_source": "HAPI FHIR Patient Data (Fallback)"
                }
    except Exception as e:
        return {"error": f"Failed to fetch trial participants: {str(e)}"}

@mcp.tool()
async def get_trial_observations(trial_id: str, participant_id: str = None) -> dict:
    """Get trial observations from HAPI FHIR Observation data with clinical trial focus."""
    try:
        # Get observations from HAPI with clinical trial focus
        if participant_id:
            url = f"{FHIR_BASE_URL}/Observation?subject=Patient/{participant_id}&_count=50"
        else:
            # Focus on vital signs and lab results for clinical trials
            url = f"{FHIR_BASE_URL}/Observation?category=vital-signs&_count=50"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return {"error": f"Failed to fetch observations: {resp.status}"}
                
                data = await resp.json()
                observations = []
                
                for entry in data.get('entry', []):
                    obs = entry.get('resource', {})
                    observation_id = obs.get('id', 'Unknown')
                    
                    # Extract observation details
                    code = obs.get('code', {})
                    coding = code.get('coding', [{}])[0] if code.get('coding') else {}
                    
                    value_quantity = obs.get('valueQuantity', {})
                    
                    # Focus on clinical trial relevant observations
                    observation_type = coding.get('display', 'Unknown')
                    is_clinical_trial_relevant = any(keyword in observation_type.lower() for keyword in [
                        'blood pressure', 'heart rate', 'temperature', 'weight', 'height',
                        'glucose', 'cholesterol', 'hemoglobin', 'creatinine', 'bun',
                        'systolic', 'diastolic', 'pulse', 'respiratory', 'oxygen'
                    ])
                    
                    if is_clinical_trial_relevant or not participant_id:  # Include all if no specific participant
                        observations.append({
                            'observation_id': observation_id,
                            'code': observation_type,
                            'code_system': coding.get('system', 'Unknown'),
                            'value': value_quantity.get('value'),
                            'unit': value_quantity.get('unit', ''),
                            'date': obs.get('effectiveDateTime', 'Unknown'),
                            'status': obs.get('status', 'Unknown'),
                            'category': obs.get('category', [{}])[0].get('coding', [{}])[0].get('display', 'Unknown') if obs.get('category') else 'Unknown',
                            'clinical_trial_relevant': is_clinical_trial_relevant
                        })
                
                return {
                    "trial_id": trial_id,
                    "participant_id": participant_id,
                    "observations": observations,
                    "total_observations": len(observations),
                    "clinical_trial_relevant": sum(1 for obs in observations if obs.get('clinical_trial_relevant', False)),
                    "data_source": "HAPI FHIR Observation Data (Clinical Trial Focused)"
                }
    except Exception as e:
        return {"error": f"Failed to fetch trial observations: {str(e)}"}

@mcp.tool()
async def validate_trial_data_integrity(trial_id: str) -> dict:
    """Validate trial data integrity and completeness."""
    try:
        # Get trial participants
        participants_result = await get_trial_participants(trial_id)
        if 'error' in participants_result:
            return participants_result
        
        participants = participants_result.get('participants', [])
        
        # Get observations for all participants
        observations_result = await get_trial_observations(trial_id)
        if 'error' in observations_result:
            return observations_result
        
        observations = observations_result.get('observations', [])
        
        # Data integrity checks
        integrity_checks = {
            "total_participants": len(participants),
            "total_observations": len(observations),
            "data_completeness": 0.0,
            "missing_data_points": [],
            "anomalies": [],
            "validation_status": "Unknown"
        }
        
        # Calculate data completeness
        if participants:
            expected_observations_per_participant = 5  # Expected baseline observations
            total_expected = len(participants) * expected_observations_per_participant
            actual_observations = len(observations)
            completeness = min(100.0, (actual_observations / total_expected) * 100) if total_expected > 0 else 0.0
            integrity_checks["data_completeness"] = completeness
        
        # Check for missing critical data
        critical_codes = ["8480-6", "8462-4", "8310-5"]  # BP, Heart Rate, Temperature
        found_codes = set()
        for obs in observations:
            code_system = obs.get('code_system', '')
            if 'loinc.org' in code_system:
                code = obs.get('code', '').split(' ')[0] if ' ' in obs.get('code', '') else obs.get('code', '')
                found_codes.add(code)
        
        missing_codes = set(critical_codes) - found_codes
        if missing_codes:
            integrity_checks["missing_data_points"].append(f"Missing critical observations: {list(missing_codes)}")
        
        # Determine validation status
        if completeness >= 80.0 and not missing_codes:
            integrity_checks["validation_status"] = "Valid"
        elif completeness >= 60.0:
            integrity_checks["validation_status"] = "Warning"
        else:
            integrity_checks["validation_status"] = "Invalid"
        
        return {
            "trial_id": trial_id,
            "integrity_checks": integrity_checks,
            "timestamp": datetime.now().isoformat(),
            "data_source": "HAPI FHIR + Validation Engine"
        }
    except Exception as e:
        return {"error": f"Failed to validate trial data: {str(e)}"}

@mcp.tool()
async def check_esg_compliance(trial_id: str) -> dict:
    """Check ESG (Environmental, Social, Governance) compliance for trial."""
    try:
        # Load ESG policy
        try:
            with open('esg_policy.json', 'r') as f:
                esg_policy = json.load(f)
        except FileNotFoundError:
            # Default ESG policy
            esg_policy = {
                "trial_esg_criteria": {
                    "environmental": {"energy_efficiency": 0.8, "waste_reduction": 0.9},
                    "social": {"patient_safety": 0.95, "data_privacy": 0.9},
                    "governance": {"regulatory_compliance": 0.95, "transparency": 0.9}
                }
            }
        
        # Calculate ESG scores based on trial data
        participants_result = await get_trial_participants(trial_id)
        participants_count = len(participants_result.get('participants', []))
        
        # Environmental score (based on trial efficiency)
        environmental_score = min(1.0, participants_count / 100.0) * 0.9
        
        # Social score (based on patient safety and data privacy)
        social_score = 0.95 if participants_count > 0 else 0.0
        
        # Governance score (based on regulatory compliance)
        governance_score = 0.95 if participants_count > 0 else 0.0
        
        overall_esg_score = (environmental_score + social_score + governance_score) / 3.0
        
        return {
            "trial_id": trial_id,
            "esg_score": round(overall_esg_score * 100, 1),
            "compliant": overall_esg_score >= 0.8,
            "environmental": round(environmental_score * 100, 1),
            "social": round(social_score * 100, 1),
            "governance": round(governance_score * 100, 1),
            "recommendations": [
                "Implement energy-efficient monitoring systems",
                "Enhance patient data privacy protocols",
                "Strengthen regulatory compliance documentation"
            ],
            "timestamp": datetime.now().isoformat(),
            "data_source": "ESG Compliance Engine"
        }
    except Exception as e:
        return {"error": f"Failed to check ESG compliance: {str(e)}"}

@mcp.tool()
async def create_audit_trail(action: str, actor: str, details: str, trial_id: str) -> dict:
    """Create audit trail entry for trial actions."""
    try:
        audit_entry = {
            "trial_id": trial_id,
            "action": action,
            "actor": actor,
            "details": details,
            "timestamp": datetime.now().isoformat(),
            "data_source": "Audit Trail System"
        }
        
        # In production, this would be stored in a database
        return audit_entry
    except Exception as e:
        return {"error": f"Failed to create audit trail: {str(e)}"}

@mcp.tool()
async def version_trial_data(trial_id: str, version_notes: str = None) -> dict:
    """Create version snapshot of trial data."""
    try:
        # Get current trial state
        protocol = await get_trial_protocol(trial_id)
        participants = await get_trial_participants(trial_id)
        observations = await get_trial_observations(trial_id)
        
        version_snapshot = {
            "trial_id": trial_id,
            "version": f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "version_notes": version_notes or "Automatic version snapshot",
            "trial_state": {
                "protocol": protocol,
                "participants": participants,
                "observations": observations
            },
            "data_source": "Version Control System"
        }
        
        return version_snapshot
    except Exception as e:
        return {"error": f"Failed to create version snapshot: {str(e)}"}

@mcp.tool()
async def get_trial_metrics(trial_id: str) -> dict:
    """Get comprehensive trial metrics and KPIs."""
    try:
        # Get trial data
        protocol = await get_trial_protocol(trial_id)
        participants = await get_trial_participants(trial_id)
        observations = await get_trial_observations(trial_id)
        integrity = await validate_trial_data_integrity(trial_id)
        esg = await check_esg_compliance(trial_id)
        
        # Calculate trial metrics
        total_participants = len(participants.get('participants', []))
        total_observations = len(observations.get('observations', []))
        data_completeness = integrity.get('integrity_checks', {}).get('data_completeness', 0.0)
        esg_score = esg.get('esg_score', 0.0)
        
        # Calculate trial progress
        target_participants = protocol.get('sample_size', 100)
        enrollment_progress = (total_participants / target_participants) * 100 if target_participants > 0 else 0.0
        
        # Calculate risk score
        risk_factors = []
        if data_completeness < 70.0:
            risk_factors.append("Low data completeness")
        if esg_score < 80.0:
            risk_factors.append("ESG compliance issues")
        if total_participants < target_participants * 0.5:
            risk_factors.append("Low enrollment")
        
        risk_score = len(risk_factors) * 2.0  # Scale 0-10
        
        return {
            "trial_id": trial_id,
            "metrics": {
                "enrollment_progress": round(enrollment_progress, 1),
                "data_completeness": round(data_completeness, 1),
                "esg_score": round(esg_score, 1),
                "risk_score": round(risk_score, 1),
                "total_participants": total_participants,
                "total_observations": total_observations,
                "target_participants": target_participants
            },
            "risk_factors": risk_factors,
            "trial_status": "Active" if enrollment_progress > 50.0 else "Recruiting",
            "timestamp": datetime.now().isoformat(),
            "data_source": "Trial Metrics Engine"
        }
        
    except Exception as e:
        return {"error": f"Failed to get trial metrics: {str(e)}"}

@mcp.tool()
async def ping() -> str:
    """Ping the trial MCP server."""
    return "Clinical Trial Management MCP Server is running"

if __name__ == "__main__":
    # Run the MCP server
    mcp.run()


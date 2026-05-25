package google

import "google.golang.org/genai"

// ToolDef is a provider-agnostic tool definition passed from the root package.
type ToolDef struct {
	Name        string
	Description string
	Parameters  map[string]ParamDef
}

// ParamDef is a provider-agnostic parameter definition for tool schemas.
type ParamDef struct {
	Type        string
	Description string
	Required    bool
	Enum        []string
	Items       *ParamDef
	Properties  map[string]ParamDef
}

// BuildTools converts provider-agnostic ToolDefs into Google GenAI tool declarations.
func BuildTools(defs []ToolDef) []*genai.Tool {
	if len(defs) == 0 {
		return nil
	}
	var declarations []*genai.FunctionDeclaration
	for _, d := range defs {
		fd := &genai.FunctionDeclaration{
			Name:        d.Name,
			Description: d.Description,
		}
		if len(d.Parameters) > 0 {
			fd.Parameters = paramsToSchema(d.Parameters)
		}
		declarations = append(declarations, fd)
	}
	return []*genai.Tool{{FunctionDeclarations: declarations}}
}

func paramsToSchema(params map[string]ParamDef) *genai.Schema {
	schema := &genai.Schema{
		Type:       genai.TypeObject,
		Properties: make(map[string]*genai.Schema),
	}
	for name, p := range params {
		schema.Properties[name] = paramToSchema(p)
		if p.Required {
			schema.Required = append(schema.Required, name)
		}
	}
	return schema
}

func paramToSchema(p ParamDef) *genai.Schema {
	s := &genai.Schema{
		Description: p.Description,
		Enum:        p.Enum,
	}
	switch p.Type {
	case "string":
		s.Type = genai.TypeString
	case "number":
		s.Type = genai.TypeNumber
	case "integer":
		s.Type = genai.TypeInteger
	case "boolean":
		s.Type = genai.TypeBoolean
	case "array":
		s.Type = genai.TypeArray
		if p.Items != nil {
			s.Items = paramToSchema(*p.Items)
		}
	case "object":
		s.Type = genai.TypeObject
		if len(p.Properties) > 0 {
			s.Properties = make(map[string]*genai.Schema)
			for name, prop := range p.Properties {
				s.Properties[name] = paramToSchema(prop)
			}
		}
	}
	return s
}


THREE.CustomShader = {

	uniforms: {
		"u_image": { type: "t", value: null },
		"u_resolution":  { type: "v2", value: null }
	},

	attributes: {
		'a_position' : { type: "f", value: null },
		'a_texCoord' : { type: "f", value: null }
	},

	vertexShader: [
'attribute vec2 a_position;',
'attribute vec2 a_texCoord;',
'uniform vec2 u_resolution;',
'varying vec2 v_texCoord;',
'void main() {',
'   vec2 zeroToOne = a_position / u_resolution;',
'   vec2 zeroToTwo = zeroToOne * 2.0;',
'   vec2 clipSpace = zeroToTwo - 1.0;',
'   gl_Position = vec4(clipSpace * vec2(1, -1), 0, 1);',
'   v_texCoord = a_texCoord;',
'}'
	].join("\n"),

	fragmentShader: [
'precision mediump float;',
'uniform sampler2D u_image;',
'varying vec2 v_texCoord;',
'void main() {',
'   gl_FragColor = texture2D(u_image, v_texCoord).bgra;',
'}'
	].join("\n")

};

THREE.FishEyeShader = {
	vertexShader : [
'uniform mat4 gxl3d_ModelViewProjectionMatrix;',
'void main() {',
'  gl_Position = gxl3d_ModelViewProjectionMatrix * gl_Vertex;',
'}'
	].join('\n'),
	fragmentShader: [
'uniform sampler2D tex0;',
'uniform vec2 resolution;',
'uniform vec2 mouse;',
'uniform float lensSize; // 0.4',
'void main(void)',
'{',
'    vec2 p = gl_FragCoord.xy / resolution.xy;',
'    vec2 m = mouse.xy / resolution.xy;',
'    vec2 d = p - m;',
'    float r = sqrt(dot(d, d)); // distance of pixel from mouse',
'    vec2 uv;',
'    vec3 col = vec3(0.0, 0.0, 0.0);',
'    if (r > lensSize+0.01) ',
'    {',
'        uv = p;',
'        col = texture2D(tex0, vec2(uv.x, -uv.y)).xyz;',
'    } ',
'    else if (r < lensSize-0.01) ',
'    {',
'        uv = m + d * r; // a.k.a. m + normalize(d) * r * r',
'        col = texture2D(tex0, vec2(uv.x, -uv.y)).xyz;',
'    }',
'    gl_FragColor = vec4(col, 1.0);',
'}',
	].join('\n'),
}
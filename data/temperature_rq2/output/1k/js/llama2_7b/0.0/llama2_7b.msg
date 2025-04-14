Fixing unused property in trackCore.js
Fixing bug in ConnectionManager._processNext() by removing unnecessary call to _cleanup()
Added Transparent button type for begin and end buttons in ResponsivePopover.js
Update hand controls component
Added constants definition for Rollup plugin
Fixed error in inventory report generation for empty end date before start date
Added exit function for NodeJS host
Added error handling for destroying generic role
Added replace() function to third-party.js to handle query parameters in URLs
Added template error configuration for webfont task
Resolve v-model config for improved performance
Fixed IE9 compatibility issue in dirtyforms.js
Added Windows 8 command proxy for Accelerometer
Normalize entrypoints for Webpack 4 compatibility
Added Errors module for handling KeyNotFoundError
Update require() to use relative path for ui-layout module
Added validationMessage cache for faster lookups
Fixed issue with key generation for macOS and Windows
Added UUID to file uploads
Fixing bug in peity.js (#100644): Changing `return $(this).each()` to `return this.each()` for better performance and compatibility.
Fixed hackish reload of iframe in templates.js
Fixed call site tracing for asyncFunctionInitialized() by removing unnecessary entries
Update animation type for picker modal from 'slide' to 'fade'
Fixing gradient filter in index.js
Added keyDown event listener for saving selected value when Enter key is pressed
Update datatable.js to improve performance by filtering options more efficiently
Added data-id attribute to regions element for better tracking and identification of regions in the waveform.
Added option to ShellCraft prototype
Update IconButton to use cloned SvgIcon element
Fixed bug related to Delta position transformation
Minify optimization: removed unused selectors optimizer
Added multiline optimization for TextMetrics class
Added parameter 'value' to pointerMove method
Added check for package contents existence in AppPermissions
Fixed push payload and options validation
Added new agent functionality
Added update interval for network and browser refresh in script.js
Added optional second argument to `executeScript` function to support non-path types
Fixed typo in PlaybackState.js\n\nDiff changes:\n- Changed line 457, column 7 from 'test' to 'test.export()'
Update exec.js to use `node` instead of './node_modules/.bin/webpack' for Webpack command
Added polling interval for Windows
Added utilities to index.js
Update physics engine initialization
Added Redis database password option to queue configuration
Added colored output option to index.js
Added configuration for bundle production
Added server and port information to client.js
Fix for RangeSlider issue with _updateHandle and live change event
Added setCustomMarker() in b/directives/custom-marker.js
Added support for Ionicons icon family in Icon.js
Updating jest configuration file to use new root directory
Fix for issue #12345: corrected pool destroy method to include callback function
Added height calculation for slideshow items in content.js
Added new device with Zigbee model and description
Added confirm button text for publish flow modals
Added method for setting ancestor and pushing child element
Added symbols for Windows platforms
Fixing null value handling in observable.js
Added support for consuming bytes after reading frame
Added configuration file parsing
Fixing default interface routing with sudo
Fixing Redback constructor to accept client object instead of hash
Added disabled attribute to SS link button based on node name
Renegotiating connection due to request from initiator
Added property 'validationObject' with deprecated alias 'validationModel' until version 4.0.0
Added require for CSSInlineEditor-test.js and updated LiveDevelopment-test.js
Added support for pointer events on touch-enabled devices
Update API to use 'chrome[eventName]' instead of 'chrome[domainName][event.name]' for event handling
Modified test/test_modrewrite.js to add describe('grunt-contrib-connect-modrewrite') and remove describe('Authentication')
Added support for processing multiple files at once
Added support for invoking generators on Chance instance
Added support for line breaks in paragraph tokens
Improve performance by removing unnecessary compare function call
Added check for invalid guide data and updated rotation property handling in MotionGuidePlugin
Fixed read() call in syncer.test.js
Reorganized highlighting logic for improved performance
Added exception for keyboard click events and Ionic app taps
Update context utils to filter out null or undefined $vnode properties
Added import of connections and updated reducer for queue selection
Update WizardProgressNavigator: Fix zero-based indexing for currentStep and oldStep
Updated common.js to add new dynamic table rows
Reverted changes to typography theme Irving
Fixing resolved path in index.js to use `__filename` instead of `__dirname` for better compatibility.
Update SIFTSCIENCE_URL to use new CDN URL
Added support for before option in VerifyStream
Fixed typo in file 'source-minified/smart.elements.js'
Update to use relative path for level2 core module
Fixing text draw issue
Added cache whitelist check for TritonAgent
Fix server error message in NylasConnection class
Added language ID to NullLanguageMode
Added warning for missing dependencies in node_modules
Added IamService dependency in ioc_module.js
Added support for multiple scopes in authentication flow
Fixed error handling in sendMessage function
Revised styles.js to remove unused color property and add letterSpacingVariants for condensed and expanded states
Added dependency on uWebSockets.js (version number now includes the dot).
Added cooldownTimers array and spaces variable for plotly user credentials in StatsPorn module
Added captureStackTrace to StatusCodeError
Added test for respecting the order of StyledComponent creation for CSS ordering
Fixing issue with visit and visitor tokens
ResolveRef: improved index ref handling
Added default value for `a` in `u.withDefault()`
Updated messages.js to reflect changes in conversation timestamp and last message
Update utils.js with HTTPS support
Fix highlight language detection for improved accuracy
Added check for file entries in onDragOver
Fixed test for ProgressBar.tick() with mock.calls.length equal to module count
Update to use waitFor instead of w for asynchronous tests
Added Windows Phone platform support for file:// remote files
Added file path to index.js
Added Thai language translation to the designerCtrl controller
Added length of values array to Statistics class
Resolve inject: fix for #6574
Added underscore dependency for interact.js
Added new whitelist for lint-bare-strings rule
Added support for retrying operations on new servers
Update ConvertArrayBuffer to use Buffer.from() instead of manual buffer creation and view manipulation
Limiting the number of offers generated for WebSocketTracker.prototype.announce() to improve performance
Updating rollup configuration to use ESM format
Update exchange data for Gateio
Added support for querying fields with underscores in their names
Added ISO 8601 timestamp to featured playlist query
Updating timedelta calculation in bitstamp.js
Added log statement to check if build mode is enabled before running tests
Updating network.js with changes related to AA address handling
Added new method for picking values from an object (#12345).
Added two new bench fixtures to the suite
Added import for rimraf module
Reformatted error messages for better readability
Added return statement at end of function
Added prevention of taps on anchors and images in slideshow.js
Added special case for parsing root password options in the form of old:oldRootPassword,new:newRootPassword
Added check for map div existence before destroying extent rectangle and rect events
Added support for site topic pages in Jetpack Connect
Fixed bug in createCacheKeyFunction.js by removing unnecessary import of path module
Update test.js to handle order book for CoinMarketCap
Added test for custom props
Added comment block for required modules
Update middleware to remove unnecessary `if` statement
Fixing typo in nconf/stores/redis test
Added emit() method for error handling in Zookeeper.prototype.listBrokers
Added globals variable to Runner constructor
Added SvnLibraryNode instance constructor
Deploying Polymath contracts on development network
Fixing line selection issue in CodeMirror.js
Added object-assign() to actions.js
Fixed new line at end of file
Added protection for inline HTML code with double curly braces and percent signs
Fixed frozen object support in traverse function
Fixing invalid event type check on target element
Added support for --help flag in generate-choco-package.js
Added support for URLs without a protocol (#12345)
Updated exchanges in ccxt.js
Fixed entity key in processor.js
Added padding properties for MJMLColumnElement component
Added support for multiple PTZ scopes
Added interval logging of peer statistics
Added a new function for handling TCP information
Updating redux types to match Flow v0.134.x-
Added return statement for `copyInfo` function
Updating dependencies for canary and beta releases
Added check for missing information grid in wordcloud2.js
Update i18n/en-us.js with changes from previous version (diff)
Reverted changes to unstackable prop
Fixing browser compatibility issue in highlighter plugin
Update index.js to use join() instead of path.normalize() for better performance and consistency
Added ChromeHeadless as a build option for Karma
Added a map for storing relative paths for context dirs
Update utils.js with improved contract and param formatting
Fixing typo in SelectDirective component
Update base64.js with internal helpers
Added event listener for DOMNodeInserted to update anchors
Migrated logs from 'nodejitsu-api' to 'preacher-api'
Added lastUpdateTime calculation for ReplSet instances
Update solhint configuration
Update util.js to simplify the logic for determining if a route is active
Added code validation for fetchLedgers
Update GuildDelete action to delete channels and voice connection upon guild deletion
Revised navigation drawer code for better performance
Added multiline tag indent past tag option
Removed unnecessary call to connection.end()
Added deleted property to Message class
Added 'initial' input type to index.js
Disabled eslintrc.js rules for node/no-extraneous-import, node/no-unpublished-import, and node/no-unsupported-features/es-syntax to improve code quality.
Added 'timeInForce' property to bcex class
Uncurrying `function-uncurry-this.js` to fix buggy `callBind` method
Added config variable to browser-index.js
Fixed error handling in retrieveData() by adding an error message to the UI
Update library path in H5P.getLibraryPath() to use H5PIntegration.url instead of H5PIntegration.libraryUrl
Added constants and documentLoader modules
Added support for blockquotes in Utility.split
Fixed keyboard event handling for better accessibility
Updates to UI Form.js based on RDF Export Form changes
Added functionality to task.js
Upsampling line width for improved performance
Reverted fixture changes
Update to use Web API's indexOf method instead of lodash's
Added custom options for sails-linker in Gruntfile
Fixing bug in setPosition() by handling undefined newPosition value
Update SimpleFormIterator.js with new fields
Added is-zeit-world utility function
Added detection for Google Calendar ID in gcal.js
Fixing offset calculation for virtual scroll\n\nThe `offset` variable was incorrect in the previous version of the code. This commit updates the calculation of `offset` to use the correct value based on the `listOffset` and `scrollDetails.scrollStart` properties.
Added new roles to pending list
Updated hot.js with 100644 Git hash
Reverted changes in defaultTheme.js
Added gain and pan properties to Sampler class
Added LRU cache for known grouping keys
Update Settings.js to use get() function with optional default value
Update spec/cwrcgit.js to use config.js instead of config.json
Added saveAssociations option to persist associations after saving instance
Rewrote store on run all tests
Added test for accepting sistyl instances in .set()
Updates to Gulpfile.js for better compatibility with CapNP-TS and TSLint
Improve stack trace prettification (#506-16) {callee, notes} -> {callee, notes, file, line, col}
Update core.js to use new URL constructor for setting href
Added function for converting objects to strings
Update ImageryLayerCatalogItem opacity property from String to Number
Updated copyright notice to include Effector Core team
Fixed bug in CWD handling
Update GuildChannel with NSFW property
Added support for scoped CSS classes in non-JS files
Fixed issue with empty file uploads in Express File Upload module
Improve heuristic for finding first focusable element
Update SocketId: Fix for device not found or token not valid issue
Added import for api module
Fixed typo in range.js
Fixed bug in isDevelopingAddon() function
Fixing issue with Screen Reader announcement of fragments
Added support for eliding unnecessary import statements
Added timeout for client requests in crawler.js
Added new spec.after implementation for function calls with length > 0
Added functionality to set the `fnc` property of the transfer element in the edit frame
Updating core.js with new event bindings and image types
Update to use `window.performance.now()` for current timestamp instead of `window.performance.webkitNow()` for compatibility with modern browsers.
Fixing IE scroll issue when focusing cell
Added transaction ID generation for send method in Connection class
Added stack trace logging for long stack traces
Moved node inside root container and updated archived status.
Added resource file 'resources/linux/snap/snapcraft.yaml' to the list of files to exclude from the build process
Added intl mock to notification spec test
Added renderer function for domain combo
Added credentials and headers for request
Added clearInternal method for relations
Added timeframes for 1m, 5m, 15m, and 4h durations
Added hasFired method to preserve pending callbacks and clear fired status
Added cluster name and size parameters to retrieve_and_rank function
Update channel mapping file
Fixing bug in View.js by returning computed value for prop 'path' instead of cloning it every time
Added getFormattedCode method to EventListenerView
Added new test cases for Pusher constructor with no cluster supplied
Fixed bug related to query options closing when multiple documents are selected in Nightwatch tests.
Added bearings parameter to route request
Fixed special char and word replacement in movies.js
Added support for multiple input values in Condition expressions
Added proxy option to WooCommerceAPI
Added algorithm information for public key getFingerprint method
Update index.js to close the WebSocket connection with a specific error code (1008) when an error occurs on the stream
Sanitizing URLs in index.js
Fixing percentage hook to use reference window for visibility calculation
Fix for async/sync engine issue in cq
Updating types.js to use block statements
Fixed navigation back function in BrowserHistory
Update find.js to use node.path instead of root.path
Added check for right-click event button
Added implication symbol handling in Parser.js
Added support for object types in `nwmatcher.js`\n\nThe `toLowerCase()` method was added to the `attribute` variable to handle case-insensitive matching.\n\nThe `if` statement was added to check if `node[attribute]` is an object, and if so, return the value of the `value` property or an empty string. Otherwise, the `getAttribute()` method is called on the node to retrieve the value of the `type` attribute or an empty string.
Fix for Crosswalk version compatibility
Updating boltrc.js to use console-based schema error reporting instead of CLI-based
Added support for binding multiple listeners in one go
Fixes issue with non-Host handshakes
Added handleActions and handleEffects methods to Account component
Added support for dest file path in buildFileList function
Update backend adaptor to use new hostname and listen port
Added ariaDescribedBy property to ResponsivePopover.js
Updated login spec with new user creation and login functionality
Added seek bar to HTML5 audio player
Added rename function to index.js
Added $http dependency for ProfileService factory
Fixing tag detection for Trumbowyg editor

Added option to pass through thenable objects
Added support for custom schemes with privileges in main.js
Added 'target' property to ConfigParser.prototype
Fixing click-awayable mixin to properly unbind click-away event on componentWillUnmount
Updating katello_object.js to fix focusout event handler for label input field
Adjusted timeout value for crawler reliability test
Added reset files list functionality in Controller.js
Added validation for RenderMode's gl program
Added check for localStorage before configuring debug mode
Added margin to barchart.js
Update seeders index for torrent {infoHash}: from {oldMaxSeeders} to {newMaxSeeders}
Update vCard.js to use lines[0] and lines[lines.length-1] instead of .shift() and .pop() for better error handling
Sorted and filtered emissions data for better performance
Added startX property to SwipeableViews component for better performance when swiping
Fix for bug in generate-icons.js
Added support for specifying mime type in form encoding
Fixing quiet time for dropdown directive
Added precision handling for money mask
Added texture loading logic for raster style
Improved fetch and clone performance by adding a small delay
Fixed issue with file addition
Fixing legacy global material IDs in Normalize.js
Adding vendor file for Ember-Native Dom Event Dispatcher (#100644)
Fixed dashboard names to use key property instead of direct reference
Fixed save method to include file name in success callback
Added computed property for enabled state in ui-slider.js
Update Bcrypt driver to use default round of 10
Uploaded progress bar directive updated to include error callback function
Added served option for materialize.js in karma.conf.js
Added test for layouting a short cycle
Added production mode default for ParsoidService
Increased maximum open files limit from 256 to 1024
Added isArray method to MapTransformer
Fixed placeholder function in jQueryBuilder test file
Fix for issue 1029: WARNING: This event won't fire if ANY extension fails to load or throws an error during init.
Update EnvStressTest.test.js with new test repeats and concurrency limits
Added check for in-memory SQLite3 pools with more than 1 connection
Added Travis CI integration
Fixing truncated keys in UrlEncoded.write() method
Fixing time format in calendar.js
Added test for proper display of FormControl size
Added test for --port being taken by another process
Added focus to input element after togglePasswordVisibility
Added copy of non-minified substance file into test folder in make.js
Invert function in Filters prototype to use div element instead of p
Added conversion of keys to an array before removing them in IterableMechanism.prototype.clear() to ensure stable iteration and avoid skipping keys or terminating early.
Update script.js with closure and optional triggers
Fixed bug in resolve function by updating data field with new data
Added Expires header to cache response to simulate a 50ms delay in CloudFlare
parseHeaders: updated emoji parsing and removed markdown token
Fixed import for esprima and added resolveBareSpecifiers and addTSExtensionToImportPath functions
Fixing key path in Index prototype
Added test for leading zero factional value after decimal
Fixing typo in Collection.removeTransform() and updating documentation
Update translations for $i18next provider
Added setter for key in core.js
Added test for route matching '/:path' not to match /
Added validation for missing parameters in `validate.js`
Updating HTML5Audio.js to use options object for src property instead of hardcoding it
Added PayPal Checkout Ready functionality
Update damage calculation for melee attacks
Added padding to bottom and top of input field in platform-input.native.js to improve user experience on non-iOS devices (#1234).
Added Polyfills for nub(), union(), and intersect() methods
Added support for nested objects and arrays in IBase.js
Added flush method to read buffer in pipe test
Added support for multiple authentication strategies and failure redirect
Added support for loading children nodes with incomplete data
Added afterEach hook for stub resolution
Added test for `each` method to handle connection death
Added tests for changing foreground and background color via string parameters
Added `res` parameter to getEnhancedContext function
Fixed secret key in JWT verification
Fixed plural forms validation in helpers.js
Updating view.js to include non-attribute options in attributes
Added create contact menu item for B2C mode
Fixed console log for better performance
Update lightbox-src-directive.js to load images from the correct URL
Added defineProperty for length property in runtime.js
Fixed error handling in log.js
Added event listener for DOMContentLoaded in dom.js
Update StreetView location handling
Update to index.js
Added event listener for tab click to refresh pending updates
Removed some relative replace statements from go.mod
Added self reference in Drone.running() for better formatting
Added canvas2dWindowHandle property to preserve context when window handle changes
Fixing invariant errors in socketsync.js
Added new mnemonic length to 32 characters
Adjusted right position of slide to 800 in Slide.spec.js
Revised BrowserWindow spec to match new window behavior
Added prototype methods for getting view and data position of annotations
Added test for deleting a has-many association with option delete=true
Update import statements for PayPal SDK constants
Update agent.js with connection_found variable
Added wx flag to fs.open() to allow for exclusive write access
Added write() method to Socket prototype
Fixing bug in on.js
Randomize gas increase probability
Added log function to append file with current timestamp
Added event listener for checkbox click to prevent propagation
Adjusted environment configuration for platform-agnostic default file
Update PropertyStack to include detached props in layers update
Update to es5-ext library
Update QIcon.js to include new icon class names
Added new directories for migrations and models
Added filter for current section in main controller
Update navigation instruction class to remove plan property and add viewPortInstructions property
Added meta property for Rubicon bid adapter
Reorganized code for better performance and readability
Added support for RGB color mode in YeelightColor class
Added cache busting for IE/Android/iOS6
Revised find function to remove unnecessary characters
Added required prop 'apikey' for PlacesAutocomplete component
Refactoring config file path handling for better compatibility with different file extensions
Added guild ID inclusion in cooldownExclusionCheck
Update license information for Aur
Added functionality to set up form data for attachments
Added filepath field to search result object in project-text-search.js
Fixing bug in WebDriver.createSession() to properly handle onQuit callback
Update Bootstrap version to 3.3.7
Update server.js to use binary representation of IPv6 peers for non-compact responses
Added debounce delay for calc handler (#100644)
Fixed issue with recursive inline partials in test/partials-test.js
Added support for 3DES encryption
Added assumption of default 'migrations' directory if config file doesn't have 'migrationsDir' key
Update AbstractEditor to remove unused properties and improve code organization
Added null check for onlogin prop in LogInModal component
Fixed error handling in getInstagramPosts function
Fixed typos and removed unused variable in binance.js
Update LeftAndMain.js with bug fixes and improvements
Added import statement for Ember-Moment addon
Updating GMapsObject with new center coordinates and zoom level
Update ONYX version to 2.3.0-pre.11-dev
Fixing plugin globalization issue
Added support for uploading files with different names
Added HTTP event handler for function definition with key 'functionKey' in ServerlessOffline.js
Added DHT Announce Interval for secondary info hash (900000 milliseconds / 15 minutes)
Fixing typo in character definitions
Added emission of 'connection' event for WebSocket clients
Fixed bug in saveAjaxAdminWindowUsingPost() by removing unnecessary parameter
Added hook for mock-fs' open event to map mount points and clean up fd mapping
Updating inputs_spec.js to use 'control-label' instead of 'form-group label'
Added log statements for localIP and device list
Added support for arrays in storage adapter spec
Added import for Keyboard component in search/index.native.js
Update leave.js to emit 'peer:disconnected' event instead of 'peer:leave'
Added generateRailsManifest option to AssetRev constructor
Added new LAN connections for each tessel
Fix for case 2 of router.js: Bound object (could be a function) handling
Added support for transition properties in supported-value.js
React class for Index page updated
Added space after reading blank key cap
Reorganized dateCell and monthCell functions for better performance
Added test for accepting target Buffer and offset parameters in fromRaw() method
Added SVGO plugin configuration to exclude non-spritesheet SVGs icons
Added new page settings sanitization logic
Added 4% coverage (95 -> 99) and fixed 1 error
Added maximum number of modules option to webpack config
Added conditional statement to only generate links when not in Electron environment
Removed ad manager reference for video player
Added support for user guild settings in Shard class
Revised Tick component to move the text 5 pixels to the right and maintain the same y-coordinate
 updated service spec to use path.join instead of concatenating strings
Updated jQuery version in main.js
Added return statement for getWorkspaceMeta function
Added import statements for Parcel, Member, Name, Repository, and Bugs types
Added property 'style' to TextField component
Added endDragDelay method for mobile drag detection
Update HLS playback to handle DVR status and position
Added labels for accepted date, published date, received date, and revision requested date; updated person labels
Fixing inequality test in isNot() function
Added 'categoryNames' field in geojson.test.js\n\nOld value: new Set(['red', 'blue'])\nNew value: ['red', 'blue']
Removed @metadata from source file
Added support for ancestor performance restrictions in pagesCursor
Added datapoints property in ccu-connection.js
Added getName() and setName() methods to Shape class
Added Google Maps widget to search page
Fixed error message for non-existent directory
Fixed skip annotation in LabelEditSpec test
Fix tbody selector
Fixed whitespace normalization in content.js
Removed unnecessary ticker variable
Fixing broken `get` method in `Cache` class
Added token validation for SocketService service
Updated cookie law functionality in cookielaw.js
Added documentation for filterMatchMode
Update dependencies and plugins for Roc plugin test suite (#100644).\n\n* Dependency 'roc-package-webpack-web' updated from '~1.0.0' to '^1.0.0-alpha'.\n* Plugin 'roc-abstract-plugin-test' updated to require version '^1.0.0-alpha'.\n
Added null and undefined handling for sorting
Added calculation and display of polygon area in Draw.Polygon handler
Added new configuration options for format and installed modules
Updating asset RID in boom.page.js
Added children prop to Textfield component
Added source property to SuperTask constructor for better module.exports handling
Added skip test for file watching on CI
Fixing validation for create button in panel.js
Added function to retrieve number of months in a year
Added updateSelected_() method to ShakaDemoFront class
React Blessed Component: Improved event listener handling
Update getConfig function to filter out unused environment variables
Added logging for assertion attributes and variables
Added check for negative position and set it to 0 in navigate() function
Added check for unsupported hash keys in Property.Type
Added Content-Type header for styling
Added node attribute for constructor in DNA Core component
Added `transform()` method to `IgnoreFormatter` class
Updating IntlTelInputApp with new country data
Fixing typo in after hook
Added support for ES modules import of base.js
Added check for numeric values with a decimal point
Update HKP constructor to use Node.js fetch API instead of global fetch
Fixed issue with insertion of element
Added validation for package name
Removed temporary Ropsten fix
Fixed whitespace encoding/decoding in `textContent` getter/setter
Fixing focus on first input in Modal component
Updating install script with canary versioning
Added extractor for getBoundingClientRect() in textwrap.js
Added check for script parsing failure
Restore error and listening listeners after updating net.js
Fixing issue with empty lines in transaction data
Fixed error handling in engineScript.js
Added support for multiple module names in entry.js
Added middleware function to route controller
Fixed issue with newline characters in gulpfile.editor.js
Added clear method to RenderTexture class
Update Gruntfile to use new task for compiling libs
Fixed error in get_instances.php response
Update CSS styler to keep writing-mode and direction properties at the root element when inherited from a parent element (#100644).
Added jumpToAnchor function for responsive element anchoring
Added server name to configuration defaults
Added response headers dump for failed handshake
Fixed HTML escaping in Markdown.setStatic() to prevent XSS attacks
Added support for new keyboard shortcuts in `InputOption`.
Update FileTreeViewModel to handle notFullyLoaded directories
Update Lokijs.js to use Object.prototype.hasOwnProperty() for property access
Added mode option to socket connector for better compatibility with proxy modes
Update skip field to handle pagination correctly
Added withdraw function for code
Added optional parameters to Shape constructor
Updated Webpack configuration for Twitter app
Added stats object to Webpack config file
Added default renderer priority
Added test for Firefox user agent
Added token field to user object in auth/user.js
Fixing issue with dropdown visibility and class names in select2.js
Update traffic graph to show more than one node
Simplified createVirtualDOM function for improved performance
Update mediaEvents.js to remove unnecessary browser import and optimize bitrate calculation
Added closure polyfill for improved browser compatibility
Added fieldless resource support
Update kue-ui-client dependency and remove unnecessary fs module import
Fixing universal render callback to return Promise.resolve() instead of undefined for async rendering
Fixed acceptable latency threshold in topics.test.js from 150 to 300
Added typeName parameter to query method
Update compare perf chart to use new series format
Added definitions for SQRT2 and afwidth in fragment shader
Update View.js to use exec() instead of done() for better performance and consistency with other components
Added indexOf method to core.js
Added import for classnames module
Added support for FairPlay DRM in assets.js
Update reducer for test cases
Update BrowserStack tests to use latest browser versions (Chrome 54.0, Firefox 66.0, IE 11.0, Edge 17.0)
Update test/test-rosidl-message-generator.js to require all message classes
Added new page titles and locations for Dustbin examples (104,6)”, 100644
Fixing step value for progressive pipeline in Scheduler.js
Fixing bug in serve.js by adding missing file
Adding method_missing stubs for subscribers
Added new Sequelize configuration for testing purposes
Added HotModuleReplacementPlugin and removed NoErrorsPlugin for improved performance
Generate externs for tasks/generate-externs.js
Added timestamp property to message object in consume method
Added support for PNG files in Webpack configuration
Added 4 new funnies to the funnies.js file: 'Updating Updater...', 'Downloading Downloader...', 'Debugging Debugger...', and 'Reading Terms and Conditions for you.'
Fixing typo in `defaultRender` function
Added storage module for UUID generation
Added log statement for successful override of minzoom flag
Added `open` parameter to DropButton component
Added support for colorxy in devices.js
Update validateElement.js to include a new input field for customized field
Update testflow.js to remove unused console.log() and improve performance by concatenating children arrays instead of creating new objects
Added my_rate threshold to options object in es6sample.js
Added navigation options for My Modal screen
Added test for saving a model with a string id and checking if a model exists
Adjusting zip level for improved compression
Fixed bug in FormBuilder component save/update/delete/cancel events handling
Added space inside expression parentheses for better readability
Fixed bug in SceneManager.add() and SceneManager.switchTo() methods
Added recursiveDelete() method to Publish class
Update request.js to use API v3
Fixing touch end listener to properly enable mobile auto-unlock
Added import for ErrorManager and translated error messages
Update RIPEMD160 hash function in eckey.js
Fixed event listener exception logging
Added check for deleting relationships when a connection is detached
Added support for headless launches by adding custom launchers from SL_LAUNCHERS and HEADLESS_LAUNCHERS to karma.conf.js
Added new test cases for Document.js
Fixed protocol and domain in less.browser.js\n\nDiff: \n--------\nOld: \n```\n-    return href.replace(/^[a-z]+:\\/[^\\/]+/, ''); // Remove protocol & domain\n+    return href.replace(/^[a-z]+:\\/[^\\/]+/, ''); // Remove protocol & domain\n```\nNew: \n```\n-    return href.replace(/^[a-z]+:\\/[^\\/]+/, ''); // Remove protocol & domain\n+    return href.replace(/^[a-z]+:\\/[^\\/]+/, ''); // Remove protocol & domain\n```
Added mocked connection for trigger() test in channels.js
Fixed HMAC calculation in index.js
Added check for previous state of open modal before applying toggleClass
Update client plugin to use new asset module naming convention
Updating index.js to return abilities instead of calling CanService
Added test for creating content node from inside InlineUI
Updating schema.js to reflect changes in the UI structure
Update to BytePushers OOP module
Update jQuery File Upload Plugin to version 5.0.3
Added new files to the Gruntfile.js
Added new OHLCV parsing logic for bars
Added getAnnotations call with promise to handle annotations clearing
Added ignore pattern for nodemon task to exclude destination directory
Added support for multiple extensions in resolveAsFile() function
Adjusting input.js for correct scale factor
Computed function optimization
Fixing bug in InputMask component
Added useEtcSslCerts option to configure SSL certificates
Added resolution for Windows file path issues in writeStats() function
Update IGN attributions to use latest logo
Added size check for folders in GlancesPluginFolders service
Added test for ignoring full embedded entities in SirenExtension
Fixing bug in text view cell deletion
Reacted to changes in GlobalNavMenu.js
Added option for specifying public directory
Added callback argument to keys method
Added support for transactions in getMore() method
Added response class and method for JSON parsing
Added new arg for script execution in commandLine.unit.js
Added check for connectedness of master link before calling callback
Fixed bug in indexedDB.js by resolving data correctly
Update ko.js with improved formatting and consistency
Update mentions index.js to use slug instead of username for user links
Added visibility CSS for new and old elements after exploding
Update atom.js to version 0.3.1
Fixed bug in test for user stream API
Fixed bug in complete function by comparing n > 10 instead of n >= 10 for better performance
Added new tests for handling undefined values and numeric Unix timestamps as input in amDateFormat directive
Fixing issue with Modal and horizontal pan responder on SwipeableViews
Fixed plugin registration on destroy
Added check for jQuery events on tear-down of action handlers
Revised clickAndHoverHelper.js to include waitFor() before clicking
Fixed bug in entity cache by removing unnecessary array declaration
Added configuration for $resource provider in resources.module.js
Release: version {{newVersion}}
Added search ID to active call in call taker reducer
Added functionality to handle Packstation addresses
Fixed ignoring of node modules in cli-harmony.js
Fixed bug in AllLayers.js by removing unnecessary addTo call
Added filter to live reload host in index.js
Added mapper function to split() method
Added support for substitutions in index.js
Update dateTime.js with improved date formatting options (ko language only)
Update component state handling
Fixing XML namespace in objectToRpcXML() and objectToXML() methods
Added new error message for when no sites have plugins
Updated processInstance page with new child element
Preview rendering improvements: replaced <[eis]> with <eis> to fix XSLT issue
Reverted changes in viewer.js related to zooming functionality
Added interface implementation for IUnknown
Added Awesome Typescript Loader and removed AotPlugin
Added singleRun option to test configuration
Fixing binding for myFunction in core.js
Added requestRenderFrame() call for improved performance
Fixing draggable behavior for input, textarea and select elements
Fix for OTA bootloader delay
Added regular expression for tag navigation
Update dependencies and constants for Spring Boot 2.3.4.RELEASE and Liquibase 4.1.1
Added Prettier plugin and rule to eslintrc.js
Adding support for unpacking manners-bin
Added onCancelled event listener for child removals
Fixed issue with spaces array
Added test for reading a sub-directory
Fixed navigation list rendering for path {path}
Removed unused function `start`
Added new route for list inspections
Added configuration for gzip static asset serving in production environment
Reflow.js: Update to handle missing body element
Added logging for minification error
Added parent reference in scrollParent calculation
Fixed content type checking for Message.js
Added ignoreEvent method to LayerSwitcher control
Added test for setting an object to null without throwing an error in field actions spec.
Added parent method to V prototype
Ported from cartodb-postgresql and updated getDistributionType function to use data parameter
Fixing typo in Funnel constructor
Fixing tile coordinates for better performance
Added cost and fee fields to bittrex.js
Fixed newline characters in goog.crypt.stringToUtf8ByteArray() for better compatibility
Require more than one pixel movement for pointer events
Added sass-cache directory for better performance
Added socket ID to slide notes event listener
Added support for setting element values using data-* attributes
Added buildStr method for string builds
Update entry.js to use bind method for creating aliases
Added check for post visibility and user permissions before inserting mention
Fixed typo in TinyMCE initialization
Added onOpen() method to XHR prototype
CoinMarketCap: updated ticker parsing to use safeTimestamp instead of safeInteger for last_updated field
Added handling of click events on the backdrop (deprecated in favor of onClose)
Added _changeSelectedDate method to update state.selectedDate and inputValue upon date change
PubSub: Update subscription handling for better performance
Fixed error handling in registerEvents function
Added PhantomJS execution timeout
Added support for optional properties in `jstype.js`
Added program property 'prorogram' to options object
Fixing DOC_TYPE regex to include the correct start and end tags
Fixing token stub for activities.js tests
Added check for array length in file form submission
Added keep property to dropped entries in compressor
Reorganized code for better readability and maintainability
Update to use underscorejs' isSafeInteger method for checking if a number is safe and natural.
React: Update Button component
Added iOS-specific logic for Safari and In-App Browser authentication
Fixing legacy `spread` prop in `remark-lint-list-item-indent` rule
Added new spec for gulp-jasmine-browser with timeout of 10 seconds
Added listenTo for curves_subset property change
Reorganized code for improved performance
Added tests for Set prototype
Updated projectHome.js with new Knockout bindings
Fixed avoidMendelify function to correctly handle cases where a value is not eligible for Mendelification
Added test for default style language and PostCSS enablement
Added canvas and rootItem variables for improved performance
Update avoid-sha3.js to use keccak256 instead of deprecated sha3
Fixed bug in widget.treeselector.js by updating line 262,7 from - self.element().val($(this).text()); to + self.element().val($(this).parent().attr('id'));
Added new unit data structure and updated readLastUnit function to handle multiple units
Updates to KilikTable.js to fix select box issue
Fixed limitInCache option to avoid cache overflow
Added import statement for Util module
Fixed image preview directive reset issue
Added function parameter 'search' for creating queries in jquery.molgenis.xrefsearch.js
Added symbol registration for ComponentJS
Update logon.js: Fix disconnected event emission
Fixed typo in accept-encoding header\n
Added support for native classes in is_array function
Added new functionality to forward webhooks from smee.io to local development environment
Truncate filter updated
Resolved conflicts in test.conflicts.js
Added support for multiple file paths in it.js
Update to lib-js-core/src/server.js to fix a bug and improve performance
Added publicPath option to force loading of images in Chrome/FF
Fixing deadzone calculation for camera resize
Update Line Chart component to use new data prop
Added support for user agent in CustomCss
Update font family in index.js
Improved importer.js to look for style declaration in package.json
Added minLength parameter to AsyncTypeahead component
Added check for empty rDNS host and converted IP addresses to lowercase for better performance
Added props for above and id to CellCreatorButtons component
Fixing Flash support for IE11 and newer by using the `Boolean()` constructor instead of `ActiveXObject()`
Reorganized columns for better layout in sunburst chart
Fixed timeout value in PromiseEvent function to avoid rejecting promise prematurely
Added support for nickname-based command recognition
Added click event handler for QBtn
Added new language 'سرائیکی' (SKR) with ID 486 and territories '034'.
Added error handling for global and daemon errors in config reducer
Added check for current working directory in check-build script
Added aria-labelledby association to FieldBase.js
Added button state to theme.js
Added test for ignoring clicks when meta key is pressed
Update config.js with new resource prefixes for LT and API Gateway
Added configuration extension for Client options
Added function for converting CSV strings to arrays
Added public API for JSLint results
Added function _gpfRequireDocumentStack for tracking document dependencies
Added support for parsing package description as Markdown (#115)
Fixing kill signal handling in worker test
Added afterDocumentChange method to CollabSession prototype
Fixed path resolution for standalone and non-master environments
Added support for charset 'utf8'
Added Chrome console log aggregation for --multi mode
Fixing issue with deep chaining in promise implementation
Added deployAll method for simulated deployment
Added support for TV channel 7 in the skjarinn endpoint
Added check for HTTPS port in rules.js
Added property lookup for JavaScriptCodeHints Session object
Added new range parsing logic for maximum safe integer
Fixed grid invalidity error in MBTiles.prototype.getGrid
Added onEnterKeyPress prop to TelephoneInput component
Added import of navigateBack from actions/router
Added new properties to Label model
Added resource 'projects/p2/databases/d3/documents/c1-/doc1' and updated geo points.
Fixing unitty issues in TimelineSignal.js
Added findOne method for MongoDB service
Updating form-input-mixin.js with bug fixes and improvements
Added new method for shutting down worker (#100644)
Added empty elements to hide trees and actions section on tab select
Added pre-built PhantomJS binary for faster builds
Added support for true color palettes in smartFlatColor()
Added support for mixed data types in Patient schema
Fixing delay parameter in AJAX query
Added 'shoulders' key to schemaForm item in b/test/services/schema-form-test.js
Added -I option to memcached command to specify a 10-minute timeout
Added option for watch mode in grunt-webpack configuration
Added binding for content clicked event to trigger close action
Update useTransition to handle from and to props correctly
Fixed bug in bitz.js by removing unnecessary console.error() call
Moved Stamen tile URL from 'http://{s}.tile.stamen.com/{variant}/{z}/{x}/{y}.{ext}' to '//stamen-tiles-{s}.a.ssl.fastly.net/{variant}/{z}/{x}/{y}.png'.
Resolve conflicting permalink values in TemplateMap
Added test for syncing a document between PouchDB and CouchDB
Added emphasize function for background color in SnackbarContent.js
Added placeholderTextColor property to GiftedMessenger component
Fixing pagination logic for tabs controller
Added log levels for outputting to std.out() in Nxs.js
Added support for legacy views
Added import of logging module
Removed jQuery catchpaste and cookie plugins
Added function isForced to check if a path is forced in sync.js
Fixing endpoint URL in client.js
Added support for requestAnimationFrame in polyfill (#12345)
Added test for put with _deleted: true
Updates to Filter Repositories Controller in bastion_katello/app/assets/javascripts/bastion_katello/content-views/details/filters/filter-repositories.controller.js
Added new icon names for tag notifications
Prioritizing decision plugin to 90
Fixed exit code on server error
Update module exports in lib/_id.js
Added flag to force loading of UV in turbolinks_events.js
Update import statements for easier module resolution
Fixed directory contents removal issue
Fix for Chrome bug in offset() method
Added check for input fields before triggering menu toggle on '/' key press
Added Safari modifier name check for Chrome bug fix
Update admin store with new filter
Fixed challenge generator module for U2F v2 compatibility
Added random unique ID for test runner in IE < 11
Reorganized resolve configuration for better performance
Update user model to include connectSid in fetch() method
Fixed navigation links for updated state
Update Input.stories.js to use the new 'Forms' theme
Added function to modify raw theme's zIndex
Fixed handling of URLs with percent-encoded symbols in web server script
Removed unnecessary debug functionality
Revised log message for zip module creation
Added regular expression for indexing JSX classes
Fixed issue with chatroom queryAll method
Fixing bug in Comlink test
Fixing invalid JavaScript in test/invalid.js\n\nThe changes are: \n\n* Changed import of `webidl2` module from `process.env.JSCOV ? require(
Fixing navigation visibility in DiscussionPage
Added significantPrecision option to Bitfinex class
Added import for SyntaxError from parser
Added strict mode and minor code improvements
Added x-mine prefix to editor mapping
Fixing bug in node.test.js by changing line -10 from 'top' to 'style top'.
Added directory for locale files in API configuration
Update addon.js to include new imdb_id values in meta collection
Added support for objects in reconfigure function
Added default export for TAP module
Added validation for max and min dates in date.test.js
Added buffer size for hermione execution
Fixed bug in gitPlugin.js by updating the URL for computing diffs to use the Git API's `diff` endpoint instead of a custom location.
Added dash pattern option for line series
Added import of stop function from utils/event.js for better performance
Added sort() method to Keyframes class
Fixed cancelUR() call in Air.test.js to match new behavior
Added const statement for Ganache module
Fixed error handling in test.js
Deprecated query argument in createPath and createHref functions
Fixing issue with duplicate URL in short.js
Added response variable for repository details
Added tests for custom domains in teamwork
Added SSL support for Vantage server
Added support for max, min and value attributes in Element class
Update request.js to include additional headers in XHR requests
Update resolver contract address in ensutils.js
Added test for default migrations directory when no migrationsDir is specified in the config file
Added mouseout event listener to color picker dialog to restore previous hover state when user moves mouse away
Added support for XVFB and disabled GPU acceleration when running in Docker
Added canihaz() for fs.notify and findit in watch API test
Improved pathfinding algorithm with timeout support
Added export of Twig.twig property
Added strict equality check for is.equal function
Added port string to elasticsearch configuration
Added protection for private endpoints
Added test for invalid input in Merge function
Added proxy address to getCdps function signature
Fixed bug in tapJunit function by updating tst variable
Update directive.js to remove unnecessary check and use VM context instead
Added test for bugos resize value in processImage.js
Fixing typo in startFlows() method
Fix for tilelive.auto module loading issue
Added RestrictFocusToChildren component to Drawer
Added debug function for easier logging
Added device power source and save functionality
Added new line of code to include the `self` variable in the `post` request
Updating graph.js with new API
Added support for extra packages in project.js
Added support for passing a log file directly (#79)
Fixing bug in clear function
Updating Stripe API version and secret key in services/stripe.js
Update index.js to use ethUtil.sha3() for code hash calculation
Fixed issue with printing to PDF by updating the code to wait for document load before setting up the PDF layout. (#12345)
Added trailing slash check in image size handling middleware
Added support for lazy filters in compatibilityValuesRamp() function
Added deltaX and deltaY properties to mouse event object for better accuracy when tracking mouse movements
Added 'shouldUpdate' prop to NavigationBar component
Update Deserializer.js to use Object.assign() instead of manual assignment
Added support for service spec file in blueprints
Added debugger statement for playTestOrResume() method
Added import statement for `..` instead of `../lib/tinymath.js
Added support for ARMv7l architecture in electron task
Reorganized slider transitions to improve performance
Fixed issue with notifications not disappearing after clicking on a database link
Fixing escapable property in Dialog.js
Added playbackRate property and getPlaybackRate() method
Added UserDumbComponent to App
Added full-screen and has-modal-card props to modal API
Added support for Mosaic Layers
Added MAC address to Tessel's hostname in rename() method
Fixing selector.js to use modern JavaScript syntax
Added old password field to user command
Added removeFile method for file deletion
Added marginTop to Item style
Added support for creating games with SC2Map
Fixed key value in unexpected-messy test to satisfy Buffer([0x08])
Fixed HTML injection vulnerability by removing unnecessary HTML tags from response.
Fixing bug in NWMatcher.js
Improving browser LiveReload performance by only writing annotations file when there are changes
Added ReactDOM.findDOMNode() for proper rendering of frame contents
Added new modal functionality for Editor test suite
Update to angular-timeago.js to fix formatting issues
Added test for charsets handling
Update comment loader template with new globals structure
Added custom 500 error handling
Added a new function to stop drones
Update account controller to handle new password creation
Fixed invalid template resolution in Email.js
Fixing bug in Bunyan2Loggly.checkBuffer() to log multiple requests correctly
Added check for starting worker process before logging error messages
Added utility functions for converting well keys to XY coordinates
Added check for empty text changes that are only a single character long and do not move the cursor position (#2329).
Fixing error handling in superagent-promise-plugin
Fixing unnecessary output node in ImageFade node
Added test for duplicate paths in CSS import
Fixed Elasticsearch index error handling in BulkWriter.write() method
Added property type validation for entity ${ENTITY_NAME}
Added toggleUi property to index.js

Fixing compact buffer formatting issue
Added exposeWithPrefix option to configuration
Fixed buffer type for multiCheck() to use `Buffer.isBuffer()` instead of `Buffer.from()` for better performance.
Added native implementation of realpathSync and realpath methods in bootstrap.js
Fixed bug in gulp-build.js by removing unused import statement for streamify and vinyl.
Added error handling for save button
Update AutoCompleteList: fixed bug with resetting search input
Updating environment.js to remove unused code and improve performance
Added option to processCssUrls in Webpack mix configuration
Added installWorkflow for 'push-renovate-typedoc' when devDependency is present
Fixing default focus of input field in App.js
Added function for parsing cookies in API
Fix for showSpinnerTimeout not being cleared when submitting
Fixing issue with document.body not being available in non-node environments
Adjusted heights in sl-grid.js to account for Ember event watcher
Added max height calculation for modal dialog to ensure proper display on different screensizes (#1234).
Fixing year remap issue in createDate function
Added alt key check for toggle network selector
Added import for _ and React
Update VTextField: fixed bug with external value changes and lazy updates
Fixing path splitting RegExp in index.js
Added timer for polling alerts (#100644)
Added Google AdSense script for right-hand sidebar
Update game.js to use THREE instead of Math for vector calculations
Added rightHeader prop to Lane component
Fixed search query for Node.js on Google Cloud
Added support for providing own promise implementation
Added processor for packet with ID 103
Patch for LeafletPatched.js
Added blur event listener to ContentEditableBinding node
Added timestamp and datetime properties to aofex.js
Added length and md5 properties to Mongo.js file
Update geckoProfilerDefaults.js with new buffer size of 100MB
Update indices.js to remove unnecessary properties
Fixing anonymous function naming convention in instruments.js
Added isTouch variable to check for touch events
Added fallback for viewport size when none is provided
Added support for ImportDirective and StructDefinition in contract source
Added focus event listener to content window
Added 'SHUFFLE_NOREPEAT' play mode to Sonos.prototype.getPlayMode()
Fixed bug in `extract` function by adding a check for `item.data.activity` before mapping it to an array.
Added support for environment detection in Loki.prototype.copy() to ensure accurate database creation
Updating index.js to use core-js modules for Set and Map
Added new ID variable for StompClient instances (#100644)
Fixing string escape issue in index.js
Added support for CLI-Color in test_filter.js
Added elenpi.htmlParser variable and updated module.exports for HTML Parser
Added error handling for input and output streams in index.js
Added support for migrations in other databases
Added module.exports.readers and module.exports.writers properties
Added check for prerelease versions in CHANGELOG.en-US.md
Added warning for sponsors about selecting sources (#100644)
Added support for controller resolution by name
Updating jQuery.molgenis.xrefmrefsearch.js to fix minimumInputLength for MREF and XREF fields
Added 'Column' component to site index
Added ENV.locationType and updated CSP to include googleapis.com
Adjusted limit in getHistoricRates to 300 - 1 = 299
Added support for commenting out code with '#' symbol
Added functionality for second password in Blockchain Wallet
Added support for Firefox on iOS
Update get-artifact-mtimes to use process.cwd() for distribution directory
Fixed env variable in nightwatch plugin
Update to use npm registry for latest version check
Fixing asynchronous callback execution for improved performance
Fixed <!-- comment --> in index.js
Fix for timezone offset in column value formatting
Added configuration option for SVGSpriteLoader
Added support for returning the Vertex Array Object (Vao) created by the GeometrySystem
Added global document variable for DOM delegation cache
Added pastebin feature for debugging
Added call to luajs.Coroutine._remove() in resume method to properly dispose of coroutine when it is terminated
Added support for seeding the random number generator with an optional `seed` argument
Added import of vendor/ember-flexberry-data/register-version.js
Added check for 'checked' property in cloned element
Debounce function improvements
Removed unnecessary variables and calculations for improved readability
Added JSON.stringify() for data in sendWebPush function
Fixing Date.now() support for older browsers
Fixed bug in combo api enablement
Added protection for VTree in patch.js
Fixing typo in extractFilesFromHtml function to accept string argument for html parameter
Fixed error handling in EnsemblJSON.fetch() with status code greater than 300
Added column type to DDL statements
Added pixel ratio setting
Added HTML state to mode definition for HTML mixed mode
